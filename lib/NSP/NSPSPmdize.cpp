//===- NSPSpmdize.cpp - NSP SPMDization -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP SPMDization/materialization pass.
//
// This pass performs a minimal bring-up materialization for NSP multi-core
// execution.
//
// The current implementation targets the simplest sharded elementwise patterns
// (e.g. vadd expressed as linalg.generic) and rewrites them into:
//   - per-core local slices using shard.all_slice
//   - local compute (linalg.generic) on the sliced tensors
//   - optional reconstitution using shard.all_gather when a global value is
//     required (e.g. materialization into a global destination memref)
//
// Expected IR (high level):
//   - A shard.grid symbol exists in the module (e.g. @nsp).
//   - Sharding descriptors (!shard.sharding) are produced and attached/used
//     as part of the propagation/planning flow.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include <iterator>
#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/IR/BuiltinAttributes.h"

// Shard dialect ops/types.
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

namespace mlir {
namespace hexagon {

namespace {

struct NSPSpmdizePass
    : public PassWrapper<NSPSpmdizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPSpmdizePass)

  NSPSpmdizePass() : NSPSpmdizePass(/*allowCollectives=*/false) {}

  /// Constructor used by createNSPSpmdizePass(bool).
  explicit NSPSpmdizePass(bool allow)
      : PassWrapper(),
        allowCollectives(
            *this, "allow-collectives",
            llvm::cl::desc(
                "Allow NSPSpmdize to insert shard collectives (e.g. all_gather)"),
            llvm::cl::init(allow)) {}

  /// PassWrapper's default clonePass implementation relies on a copy
  /// constructor. Pass::Option is not copyable, so we must explicitly define
  /// a copy constructor that re-initializes the options from values.
  NSPSpmdizePass(const NSPSpmdizePass &other)
      : NSPSpmdizePass(static_cast<bool>(other.allowCollectives)) {}

  StringRef getArgument() const final { return "nsp-spmdize"; }

  StringRef getDescription() const final {
    return "SPMD transformation for multi-NSP execution";
  }

  // When enabled, this pass is allowed to materialize collectives (e.g.
  // shard.all_gather) as part of the bring-up SPMDization.
  Option<bool> allowCollectives;

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::scf::SCFDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();

    ModuleOp module = getOperation();

    // Validate that the expected grid symbol exists.
    // The planner currently creates shard.grid @nsp.
    auto grid = module.lookupSymbol<mlir::shard::GridOp>("nsp");
    if (!grid) {
      module.emitError()
        << "NSPSpmdizePass expected a 'shard.grid' symbol named '@nsp' "
           "in the module, but none was found. "
           "Ensure NSP shard planning ran and created the grid.";
      signalPassFailure();
      return;
    }

    // Read grid shape (assume 1D grid for now).
    auto shape = grid.getShape();
    if (shape.empty()) {
      module.emitError() << "shard.grid '@nsp' has an empty shape";
      signalPassFailure();
      return;
    }

    const int64_t numShards = shape.front();
    if (numShards <= 0) {
      module.emitError() << "invalid shard grid size (shape[0]) = "
                         << numShards;
      signalPassFailure();
      return;
    }

    // HELPER for strip shard tensor annotations inside loops that were explicitly
    // distributed by this pass (marked with 'nsp.distributed').
    //
    // Rationale:
    // In non-collective mode, the SPMD partitioning is expressed by the loop
    // schedule (lb/step adjusted using shard.process_linear_index). Any
    // shard.shard/shard.sharding inside such loops becomes stale metadata and
    // may confuse later lowerings (e.g., ShardToLLVM).
    //
    // We keep shard.grid and shard.process_linear_index, and we do NOT touch
    // shard annotations outside distributed loops.
    auto stripShardAnnotationsInDistributedLoops =
        [&](mlir::func::FuncOp func) {
          SmallVector<Operation *> eraseList;

          func.walk([&](mlir::scf::ForOp forOp) {
            if (!forOp->hasAttr("nsp.distributed"))
              return;

            Block *body = forOp.getBody();

            // 1: Replace and erase shard.shard wrappers inside the loop body.
            body->walk([&](mlir::shard::ShardOp op) {
              // shard.shard is a value wrapper: replace result with input.
              if (op->getNumOperands() < 1 || op->getNumResults() < 1)
                return;
              Value wrapped = op->getOperand(0);
              Value res = op->getResult(0);
              res.replaceAllUsesWith(wrapped);
              eraseList.push_back(op);
            });

            // 2: Erase shard.sharding descriptors inside the loop body if unused.
            // These usually become dead after removing shard.shard wrappers.
            body->walk([&](mlir::shard::ShardingOp op) {
              if (op->getNumResults() != 1)
                return;
              if (op->getResult(0).use_empty())
                eraseList.push_back(op);
            });
          });

          for (Operation *op : eraseList)
            op->erase();
        };

    // HELPER for Loop distribution (no collectives).
    //
    // For kernels with DOALL loops, sharding the reduced dimension requires
    // cross-shard reductions. In non-collective mode, the correct strategy is
    // to distribute the OUTER loop (e.g. batch) across NSPs and keep inner
    // vectors intact.
    //
    // We implement a simple cyclic distribution (block dist is a future work):
    //   original: for i = lb .. ub step s
    //   shard k : for i = lb + k*s .. ub step (s*numShards)
    //
    // This avoids any communication and preserves semantics for reductions
    // within the loop body.

    auto distributeScfForCyclic = [&](mlir::func::FuncOp func) -> LogicalResult {
      OpBuilder b(func.getContext());
      SmallVector<mlir::scf::ForOp> loops;
      func.walk([&](mlir::scf::ForOp forOp) { loops.push_back(forOp); });

      // Returns true iff `v` is derived from `iv` via a restricted set of
      // affine-like integer/index ops. This is a conservative "taint" analysis
      // used to detect whether the loop IV participates in output indexing.
      auto reachesFromIv = [&](Value iv, Value v) -> bool {
        if (v == iv)
          return true;

        // BFS over the def-use chain backwards (operand -> defining op -> its operands).
        llvm::SmallVector<Value, 16> worklist;
        llvm::SmallPtrSet<Value, 32> visited;
        worklist.push_back(v);

        auto enqueue = [&](Value x) {
          if (!x)
            return;
          if (visited.insert(x).second)
            worklist.push_back(x);
        };

        while (!worklist.empty()) {
          Value cur = worklist.pop_back_val();
          if (cur == iv)
            return true;

          Operation *def = cur.getDefiningOp();
          if (!def)
            continue;

          // Allow common integer/index plumbing ops.
          // This intentionally ignores complex control/dataflow.
          if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp,
                  arith::DivSIOp, arith::DivUIOp,
                  arith::RemSIOp, arith::RemUIOp,
                  arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
                  arith::AndIOp, arith::OrIOp, arith::XOrIOp,
                  arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                  arith::IndexCastOp, arith::IndexCastUIOp,
                  arith::ConstantOp>(def)) {
            for (Value opnd : def->getOperands())
              enqueue(opnd);
            continue;
          }

          // Allow memref.cast / view-like ops in the index path.
          if (isa<memref::CastOp>(def)) {
            for (Value opnd : def->getOperands())
              enqueue(opnd);
            continue;
          }
        }

        return false;
      };

      // Returns true iff the loop appears to perform per-iteration output
      // writes whose address depends on the IV, i.e. a DOALL-style tiled store.
      auto shouldDistributeLoop = [&](mlir::scf::ForOp forOp) -> bool {
        Value iv = forOp.getInductionVar();

        bool foundIvIndexedStore = false;

        // Direct memref.store with IV-derived indices.
        forOp.walk([&](memref::StoreOp st) {
          for (Value idx : st.getIndices()) {
            if (reachesFromIv(iv, idx)) {
              foundIvIndexedStore = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
        if (foundIvIndexedStore)
          return true;

        // bufferization.materialize_in_destination where dest is a view into the
        // output buffer whose dynamic offset is derived from the IV.
        // This matches patterns like:
        //   %sv = memref.subview %dst[%off] ...
        //   bufferization.materialize_in_destination %t, %sv
        // as well as:
        //   %rc = memref.reinterpret_cast %dst to offset: [%off], sizes: [...]
        //   bufferization.materialize_in_destination %t, %rc
        forOp.walk([&](bufferization::MaterializeInDestinationOp mat) {
          Value dest = mat->getOperand(1);

          // Strip trivial casts.
          while (auto c = dest.getDefiningOp<memref::CastOp>())
            dest = c.getSource();

          // Case 1: memref.subview with dynamic offsets.
          if (auto sub = dest.getDefiningOp<memref::SubViewOp>()) {
            for (OpFoldResult ofr : sub.getMixedOffsets()) {
              if (auto v = dyn_cast<Value>(ofr)) {
                if (reachesFromIv(iv, v)) {
                  foundIvIndexedStore = true;
                  return WalkResult::interrupt();
                }
              }
            }
            return WalkResult::advance();
          }

          // Case 2: memref.reinterpret_cast with dynamic offset/sizes/strides.
          if (auto rc = dest.getDefiningOp<memref::ReinterpretCastOp>()) {
            // Operands: source, offset, sizes..., strides...
            // Conservatively check all dynamic operands except the source.
            for (unsigned oi = 1, oe = rc->getNumOperands(); oi < oe; ++oi) {
              if (reachesFromIv(iv, rc->getOperand(oi))) {
                foundIvIndexedStore = true;
                return WalkResult::interrupt();
              }
            }
          }

          return WalkResult::advance();
        });

        return foundIvIndexedStore;
      };

      for (mlir::scf::ForOp forOp : loops) {
        // Bring-up constraints:
        //  - no iter_args / no results
        //  - induction var is index or an integer type with sufficient range
        if (!forOp.getResults().empty() || !forOp.getInitArgs().empty())
          continue;

        // Only distribute loops that look DOALL and "tile-store" safe:
        // the IV must participate in the output addressing (store/subview offset).
        if (!shouldDistributeLoop(forOp))
          continue;

        mlir::Type ivTy = forOp.getInductionVar().getType();
        bool ivIsIndex = mlir::isa<mlir::IndexType>(ivTy);
        auto ivIntTy = mlir::dyn_cast<mlir::IntegerType>(ivTy);

        if (!ivIsIndex && !ivIntTy)
          continue;

        if (ivIntTy && ivIntTy.getWidth() < 32) {
          // Avoid silent overflow for small index types (e.g., i16).
          forOp.emitRemark()
              << "NSPSpmdize: skipping scf.for distribution for small integer IV type "
              << ivTy;
          continue;
        }

        Value lb = forOp.getLowerBound();
        Value ub = forOp.getUpperBound();
        Value step = forOp.getStep();

        // Require bounds/step to match the IV type to keep the transformation simple.
        if (lb.getType() != ivTy || ub.getType() != ivTy || step.getType() != ivTy)
          continue;

        Location loc = forOp.getLoc();
        b.setInsertionPoint(forOp);

        // procIdx: index in [0, numShards)
        Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);
        // Cast procIdx to the IV type (index stays index; integer gets index_cast).
        Value procInIvTy = procIdx;
        if (!ivIsIndex)
          procInIvTy = b.create<arith::IndexCastOp>(loc, ivTy, procIdx);

        // newLb = lb + procI32 * step
        Value offset = b.create<arith::MulIOp>(loc, procInIvTy, step);
        Value newLb = b.create<arith::AddIOp>(loc, lb, offset);

        // newStep = step * numShards
        Value cNum = ivIsIndex
                         ? static_cast<Value>(b.create<arith::ConstantIndexOp>(loc, numShards))
                         : static_cast<Value>(b.create<arith::ConstantIntOp>(
                               loc, numShards, ivIntTy.getWidth()));

        Value newStep = b.create<arith::MulIOp>(loc, step, cNum);

        // Create the distributed loop.
        auto newFor = b.create<mlir::scf::ForOp>(loc, newLb, ub, newStep);
        // Mark this loop so we can precisely clean up stale shard annotations
        // only within distributed loops (non-collective mode).
        newFor->setAttr("nsp.distributed", b.getUnitAttr());

        // Clone the body operations, remapping the induction variable.
        Block *oldBody = forOp.getBody();
        Block *newBody = newFor.getBody();

        // Remove the default terminator inserted by builder.
        newBody->getOperations().clear();

        IRMapping mapping;
        mapping.map(forOp.getInductionVar(), newFor.getInductionVar());

        for (Operation &op : oldBody->without_terminator()) {
          b.setInsertionPointToEnd(newBody);
          b.clone(op, mapping);
        }
        b.setInsertionPointToEnd(newBody);
        b.create<mlir::scf::YieldOp>(loc);

        // Replace uses of the old loop results (none here) and erase old loop.
        forOp.erase();
      }
      return success();
    };

    // ---------
    // Main Body
    // ---------

    if (!allowCollectives) {
      // In non-collective mode, prioritize loop distribution. This is the
      // required strategy for patterns with intra-tile reductions (e.g. softmax).
      module.walk([&](mlir::func::FuncOp func) {
        if (failed(distributeScfForCyclic(func))) {
          signalPassFailure();
          return;
        }
      });

      // After distribution, strip shard tensor annotations only inside
      // distributed loops. Outside those loops, sharding metadata (if any)
      // is preserved.
      module.walk([&](mlir::func::FuncOp func) {
        stripShardAnnotationsInDistributedLoops(func);
      });

    }

    // Optionally sanity-check that we have some sharding descriptors.
    // This is a warning (not a hard error) because some pipelines may
    // legally run with no sharding yet (e.g. early bring-up).
    int64_t numShardingOps = 0;
    module.walk([&](mlir::shard::ShardingOp op) { ++numShardingOps; });

    if (numShardingOps == 0) {
      module.emitWarning()
          << "NSPSpmdizePass found shard.grid '@nsp' but no 'shard.sharding' "
             "ops in the module. This may be expected during early bring-up, "
             "but if it is unexpected, ensure sharding propagation created "
             "sharding descriptors.";
    }

    // Minimal bring-up materialization.
    //
    // We intentionally keep this scoped and conservative:
    //   - Only elementwise linalg.generic with identity indexing maps.
    //   - Only sharding split along axis 0.
    //   - Only ranked tensors with static sizes (for now).
    //
    // The goal is to unblock end-to-end pipeline bring-up and validate the NSP
    // sharding annotations and parameter plumbing.

    const SmallVector<mlir::shard::GridAxis> gridAxes = {
        static_cast<mlir::shard::GridAxis>(0)};
    // Some Shard ops (e.g. all_gather) expect grid_axes as i16.
    const SmallVector<int16_t> gridAxesI16 = {static_cast<int16_t>(0)};

    const int64_t splitAxis = 0;
    const llvm::APInt splitAxisAP(/*numBits=*/64, /*val=*/splitAxis,
                                  /*isSigned=*/true);

    // Helper to compute a "local" tensor type for split along axis 0.
    auto getLocalType = [&](RankedTensorType globalTy) -> RankedTensorType {
      if (!globalTy || globalTy.getRank() == 0)
        return RankedTensorType();
      int64_t dim0 = globalTy.getDimSize(0);
      if (ShapedType::isDynamic(dim0))
        return RankedTensorType();
      if (dim0 % numShards != 0)
        return RankedTensorType();
      SmallVector<int64_t> newShape(globalTy.getShape().begin(),
                                    globalTy.getShape().end());
      newShape[0] = dim0 / numShards;
      return RankedTensorType::get(newShape, globalTy.getElementType(),
                                   globalTy.getEncoding());
    };

    module.walk([&](mlir::func::FuncOp func) {
      OpBuilder b(func.getContext());

      // Find a bufferization.materialize_in_destination sink for `v` while
      // allowing a trivial chain of sharding wrappers.
      //
      // Sharding propagation often wraps values with shard.shard, so the IR can
      // look like:
      //   %r  = linalg.generic ... -> tensor<...>
      //   %r1 = shard.shard %r to %sharding
      //   bufferization.materialize_in_destination %r1 in %dst
      //
      // In store-by-tile mode, we rewrite the materialization to store the
      // local tile into a subview of %dst, and erase the wrapper chain.
      auto findMaterializeSink =
          [&](Value v)
              -> std::optional<std::pair<
                  bufferization::MaterializeInDestinationOp,
                  SmallVector<Operation *>>> {
        SmallVector<Operation *> wrappers;
        Value cur = v;

        while (true) {
          // Keep bring-up simple: require a single-use chain.
          if (!cur.hasOneUse())
            return std::nullopt;

          Operation *user = *cur.getUsers().begin();

          // Allow shard.shard wrappers.
          if (auto shardOp = dyn_cast<mlir::shard::ShardOp>(user)) {
            // Be robust to ShardOp operand naming differences across versions.
            if (shardOp->getNumOperands() < 1 || shardOp->getOperand(0) != cur)
              return std::nullopt;
            wrappers.push_back(user);
            cur = shardOp->getResult(0);
            continue;
          }

          // Allow an optional tensor.cast in between.
          if (auto castOp = dyn_cast<tensor::CastOp>(user)) {
            if (castOp.getSource() != cur)
              return std::nullopt;
            wrappers.push_back(user);
            cur = castOp.getResult();
            continue;
          }

          // Accept bufferization.materialize_in_destination.
          if (auto mat =
                  dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
            if (mat->getOperand(0) != cur)
              return std::nullopt;
            return std::make_optional(std::make_pair(mat, wrappers));
          }

          return std::nullopt;
        }
      };

      // Use a worklist since we'll rewrite in-place.
      SmallVector<mlir::linalg::GenericOp> worklist;
      func.walk([&](mlir::linalg::GenericOp g) {
        worklist.push_back(g);
      });

      for (mlir::linalg::GenericOp g : worklist) {
        // Pattern: elementwise 1D generic with identity maps.
        if (g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1)
          continue;
        if (g.getNumLoops() != 1)
          continue;
        auto iters = g.getIteratorTypesArray();
        if (iters.size() != 1 ||
            iters.front() != mlir::utils::IteratorType::parallel)
          continue;

        auto maps = g.getIndexingMapsArray();
        if (maps.size() != 3)
          continue;
        if (!maps[0].isIdentity() || !maps[1].isIdentity() ||
            !maps[2].isIdentity())
          continue;

        Value in0 = g.getDpsInputOperand(0)->get();
        Value in1 = g.getDpsInputOperand(1)->get();
        Value oldInit = g.getDpsInitOperand(0)->get();

        auto in0Ty = dyn_cast<RankedTensorType>(in0.getType());
        auto in1Ty = dyn_cast<RankedTensorType>(in1.getType());
        auto outResTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
        if (!in0Ty || !in1Ty || !outResTy)
          continue;

        // Restrict to 1D tensors for now.
        if (in0Ty.getRank() != 1 || in1Ty.getRank() != 1 ||
            outResTy.getRank() != 1)
          continue;

        auto localTy = getLocalType(outResTy);
        if (!localTy) {
          g.emitError() << "NSPSpmdize: cannot compute local tile type for "
                           "result "
                        << outResTy << " with grid size " << numShards
                        << " (requires static dim0 divisible by grid size)";
          signalPassFailure();
          return;
        }

        // Inputs must be consistently splittable.
        if (getLocalType(in0Ty) != localTy || getLocalType(in1Ty) != localTy) {
          g.emitError() << "NSPSpmdize: unsupported elementwise generic; input "
                           "types are not consistently splittable along axis 0";
          signalPassFailure();
          return;
        }

        Location loc = g.getLoc();
        b.setInsertionPoint(g);

        // Non-collective path must ONLY rewrite ops that directly materialize
        // into a destination memref. Intermediate elementwise ops (common in
        // pipelines like softmax) must remain intact; otherwise, moving the
        // region body (takeBody) would leave the original op with an empty
        // region and trigger verifier errors.
        std::optional<std::pair<bufferization::MaterializeInDestinationOp,
                                SmallVector<Operation *>>>
            sink;
        if (!allowCollectives) {
          sink = findMaterializeSink(g.getResult(0));
          if (!sink) {
            // No materialization sink => this is an intermediate tensor.
            continue;
          }
        }

        // Slice inputs into per-core tiles.
        Value in0Local = mlir::shard::AllSliceOp::create(
                           b, loc, /*result_type=*/localTy,
                           /*input=*/in0,
                           /*grid=*/"nsp",
                           /*gridAxes=*/gridAxes,
                           /*sliceAxis=*/splitAxis).getResult();
        Value in1Local = mlir::shard::AllSliceOp::create(
                           b, loc, /*result_type=*/localTy,
                           /*input=*/in1,
                           /*grid=*/"nsp",
                           /*gridAxes=*/gridAxes,
                           /*sliceAxis=*/splitAxis).getResult();

        // Create (or reuse) a local init tensor for the output.
        Value outLocalInit = oldInit;
        auto oldInitTy = dyn_cast<RankedTensorType>(oldInit.getType());
        if (!oldInitTy || oldInitTy != localTy) {
          outLocalInit = b.create<mlir::tensor::EmptyOp>(
              loc, localTy.getShape(), localTy.getElementType());
        }

        // Clone the generic op with local operands.
        auto newGeneric = b.create<mlir::linalg::GenericOp>(
            loc, /*resultTensorTypes=*/TypeRange{localTy},
            /*inputs=*/ValueRange{in0Local, in1Local},
            /*outputs=*/ValueRange{outLocalInit},
            /*indexingMaps=*/g.getIndexingMaps(),
            /*iteratorTypes=*/g.getIteratorTypes(),
            /*doc=*/nullptr, /*libraryCall=*/nullptr);

        // Move the region body (elementwise computation) over.
        newGeneric.getRegion().takeBody(g.getRegion());

        // Materialization policy:
        // -----------------------
        // *allowCollectives=true reconstitute a global tensor via
        // shard.all_gather and keep the existing destination materialization.
        //
        // *allowCollectives=false store the local tile directly into a
        // subview of the global destination (store-by-tile), avoiding communication.

        Value localResult = newGeneric.getResult(0);

        if (allowCollectives) {
          Value globalResult = mlir::shard::AllGatherOp::create(
              b, loc, /*result=*/outResTy,
              /*grid=*/"nsp",
              /*grid_axes=*/llvm::ArrayRef<int16_t>(gridAxesI16),
              /*input=*/localResult,
              /*gather_axis=*/splitAxisAP).getResult();
          g.getResult(0).replaceAllUsesWith(globalResult);
          g.erase();
          continue;
        }

        // Non-collective path: use the previously found materialization sink.
        auto mat = sink->first;
        auto &wrappers = sink->second;

        // Destination must be a 1D memref.
        Value dest = mat->getOperand(1);
        auto destTy = dyn_cast<MemRefType>(dest.getType());
        if (!destTy || destTy.getRank() != 1) {
          mat.emitError() << "NSPSpmdize: expected a 1D memref destination in "
                             "bufferization.materialize_in_destination";
          signalPassFailure();
          return;
        }

        auto localResTy = dyn_cast<RankedTensorType>(localResult.getType());
        int64_t tileSize = localResTy.getDimSize(0);
        if (ShapedType::isDynamic(tileSize)) {
          mat.emitError() << "NSPSpmdize: store-by-tile requires a static tile size";
          signalPassFailure();
          return;
        }

        // Insert offset/subview computation right before the materialization.
        b.setInsertionPoint(mat);

        // procIdx is an index in [0, numShards).
        Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);

        // offset = procIdx * tileSize
        Value cTileVal = b.create<arith::ConstantIndexOp>(loc, tileSize);
        Value offset = b.create<arith::MulIOp>(loc, procIdx, cTileVal);

        SmallVector<OpFoldResult> offsets = {offset};                  // dynamic offset
        SmallVector<OpFoldResult> sizes = {b.getIndexAttr(tileSize)};  // static size
        SmallVector<OpFoldResult> strides = {b.getIndexAttr(1)};       // static stride

        // Create a subview into the destination buffer corresponding to this
        // process' tile.
        auto subLayout = StridedLayoutAttr::get(
            destTy.getContext(), /*offset=*/ShapedType::kDynamic,
            /*strides=*/ArrayRef<int64_t>{1});
        auto subTy = MemRefType::get(
            ArrayRef<int64_t>{tileSize}, destTy.getElementType(), subLayout,
            destTy.getMemorySpace());

        Value destSubview =
            b.create<memref::SubViewOp>(loc, subTy, dest, offsets, sizes, strides);

        // Rewrite the materialization to store only this tile.
        mat->setOperand(0, localResult);
        mat->setOperand(1, destSubview);

        // Wrapper chain is now dead; erase in reverse order.
        for (Operation *op : llvm::reverse(wrappers))
          op->erase();

        // Erase the original global generic.
        g.erase();
      }
    });
  }

private:

};

} // namespace

std::unique_ptr<Pass> createNSPSpmdizePass() {
  return std::make_unique<NSPSpmdizePass>();
}

std::unique_ptr<Pass> createNSPSpmdizePass(bool allowCollectives) {
  return std::make_unique<NSPSpmdizePass>(allowCollectives);
}

} // namespace hexagon
} // namespace mlir
