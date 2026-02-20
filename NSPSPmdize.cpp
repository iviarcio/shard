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
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

// Shard dialect ops/types.
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

namespace mlir {
namespace hexagon {

namespace {

struct NSPSpmdizePass
    : public PassWrapper<NSPSpmdizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPSpmdizePass)

  NSPSpmdizePass() = default;

  explicit NSPSpmdizePass(bool allowCollectives)
    : allowCollectives(*this, "allow-collectives",
                       llvm::cl::desc("Allow NSPSpmdize to insert shard collectives (e.g. all_gather)"),
                       llvm::cl::init(false)) {
    allowCollectives = allow;
  }

  StringRef getArgument() const final { return "nsp-spmdize"; }

  StringRef getDescription() const final {
    return "SPMD transformation for multi-NSP execution";
  }

  Option<bool> allowCollectives{
      *this, "allow-collectives",
      llvm::cl::desc("Allow NSPSpmdize to insert shard collectives (e.g. all_gather)"),
      llvm::cl::init(false)};

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();
    ctx->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();

    ModuleOp module = getOperation();

    // When enabled, this pass is allowed to materialize collectives (e.g.
    // shard.all_gather) as part of the bring-up SPMDization.

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

      // Worklist since we'll rewrite in-place.
      SmallVector<mlir::linalg::GenericOp> worklist;
      func.walk([&](mlir::linalg::GenericOp g) { worklist.push_back(g); });

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

        // Slice inputs into per-core tiles.
        Value in0Local = mlir::shard::AllSliceOp::create(
                           b, loc, /*result_type=*/localTy,
                           /*input=*/in0,
                           /*grid=*/"nsp",
                           /*gridAxes=*/gridAxes,
                           /*sliceAxis=*/splitAxis)
                           .getResult();
        Value in1Local = mlir::shard::AllSliceOp::create(
                           b, loc, /*result_type=*/localTy,
                           /*input=*/in1,
                           /*grid=*/"nsp",
                           /*gridAxes=*/gridAxes,
                           /*sliceAxis=*/splitAxis)
                           .getResult();

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

        // Bring-up path: reconstitute to the original global type so existing
        // bufferization.materialize_in_destination keeps working.
        Value replacement = newGeneric.getResult(0);
        if (!allowCollectives.getValue()) {
          g.emitError() << "NSPSpmdize: rewriting requires shard.all_gather but "
                           "collectives are disabled";
          signalPassFailure();
          return;
        }
        replacement = mlir::shard::AllGatherOp::create(
                           b, loc, /*result=*/outResTy,
                           /*grid=*/"nsp",
                           /*grid_axes=*/llvm::ArrayRef<int16_t>(gridAxesI16),
                           /*input=*/replacement,
                           /*gather_axis=*/splitAxisAP).getResult();

        g.getResult(0).replaceAllUsesWith(replacement);
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

