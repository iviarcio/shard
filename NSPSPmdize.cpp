//===- NSPSpmdize.cpp - NSP SPMDization -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP SPMDization/materialization pass.
//
// This pass performs a minimal bring-up materialization for multi-NSP execution.
//
// Today it targets the simplest sharded elementwise patterns (e.g. vadd) and
// rewrites them into:
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
    : allowCollectives(allowCollectives) {}

  StringRef getArgument() const final { return "nsp-spmdize"; }

  StringRef getDescription() const final {
    return "SPMD transformation for NSP multi-core execution";
  }

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
    const int64_t splitAxis = 0;

    // Helper to compute a "local" tensor type for split along axis 0.
    auto getLocalType = [&](RankedTensorType globalTy) -> RankedTensorType {
      assert(globalTy && "expected ranked tensor type");
      SmallVector<int64_t> shape(globalTy.getShape().begin(),
                                 globalTy.getShape().end());
      if (shape.empty())
        return globalTy;
      int64_t dim0 = shape[0];
      if (ShapedType::isDynamic(dim0))
        return RankedTensorType();
      if (dim0 % numShards != 0)
        return RankedTensorType();
      shape[0] = dim0 / numShards;
      return RankedTensorType::get(shape, globalTy.getElementType(),
                                   globalTy.getEncoding());
    };

    // Rewrite matching patterns per-function.
    module.walk([&](func::FuncOp func) {
      OpBuilder b(func.getContext());

      // Use a worklist since we will mutate the IR.
      SmallVector<linalg::GenericOp> generics;
      func.walk([&](linalg::GenericOp g) { generics.push_back(g); });

      for (linalg::GenericOp g : generics) {
        // Pattern: elementwise vadd-like generic
        // 2 inputs, 1 output, 1 loop, parallel iterator, identity indexing maps
        if (g.getNumDpsInputs() != 2 || g.getNumDpsInits() != 1)
          continue;

        if (g.getNumLoops() != 1)
          continue;

        auto iters = g.getIteratorTypesArray();
        if (iters.size() != 1 || iters.front() != utils::IteratorType::parallel)
          continue;

        auto maps = g.getIndexingMapsArray();
        if (maps.size() != 3)
          continue;
        if (!maps[0].isIdentity() || !maps[1].isIdentity() ||
            !maps[2].isIdentity())
          continue;

        Value in0 = g.getDpsInputOperand(0)->get();
        Value in1 = g.getDpsInputOperand(1)->get();

        auto in0Ty = dyn_cast<RankedTensorType>(in0.getType());
        auto in1Ty = dyn_cast<RankedTensorType>(in1.getType());
        auto outResTy = dyn_cast<RankedTensorType>(g.getResult(0).getType());
        if (!in0Ty || !in1Ty || !outResTy)
          continue;

        // Only split along axis 0 for now.
        auto localResTy = getLocalType(outResTy);
        if (!localResTy) {
          g.emitError() << "NSPSpmdize: cannot compute local tile type for result "
                        << outResTy << " with grid size " << numShards
                        << " (requires static dim0 divisible by grid size)";
          signalPassFailure();
          return;
        }

        // Inputs must be compatible with the result tile type.
        auto localIn0Ty = getLocalType(in0Ty);
        auto localIn1Ty = getLocalType(in1Ty);
        if (!localIn0Ty || !localIn1Ty || localIn0Ty != localResTy ||
            localIn1Ty != localResTy) {
          g.emitError() << "NSPSpmdize: unsupported elementwise generic; input/result "
                           "types are not consistently splittable along axis 0";
          signalPassFailure();
          return;
        }

        Location loc = g.getLoc();
        b.setInsertionPoint(g);

        // Slice inputs.
        // Note: grid is referenced by symbol name ("nsp") so it remains stable
        // under cloning/moving.
        Value in0Local = b.create<mlir::shard::AllSliceOp>(
            loc, localIn0Ty, in0, /*grid=*/"nsp", gridAxes, splitAxis);
        Value in1Local = b.create<mlir::shard::AllSliceOp>(
            loc, localIn1Ty, in1, /*grid=*/"nsp", gridAxes, splitAxis);

        // Create or reuse a local init tensor for the output.
        // If a previous pass already produced a correctly-typed local init,
        // reuse it; otherwise, synthesize a fresh tensor.empty.
        Value oldInit = g.getDpsInitOperand(0)->get();
        Value outLocalInit = oldInit;
        auto oldInitTy = dyn_cast<RankedTensorType>(oldInit.getType());
        if (!oldInitTy || oldInitTy != localResTy) {
          outLocalInit = b.create<tensor::EmptyOp>(
              loc, localResTy.getShape(), localResTy.getElementType());
        }

        // Clone the generic op with local operands.
        auto newGeneric = b.create<linalg::GenericOp>(
            loc, localResTy, ValueRange{in0Local, in1Local},
            ValueRange{outLocalInit}, g.getIndexingMaps(), g.getIteratorTypes(),
            /*doc=*/nullptr, /*libraryCall=*/nullptr);

        // Move the region body (elementwise computation) over.
        newGeneric.getRegion().takeBody(g.getRegion());

        // Bring-up path: reconstitute to the original global type so existing
        // bufferization.materialize_in_destination keeps working.
        Value replacement = newGeneric.getResult(0);
        if (!allowCollectives) {
          g.emitError() << "NSPSpmdize: rewriting requires shard.all_gather but "
                           "collectives are disabled";
          signalPassFailure();
          return;
        }
        replacement = b.create<mlir::shard::AllGatherOp>(
            loc, outResTy, replacement, /*grid=*/"nsp", gridAxes, splitAxis);

        g.replaceAllUsesWith(replacement);
        g.erase();

        // NOTE: We intentionally do not attempt to clean up now-dead shard.shard
        // wrappers here. Canonicalization/CSE later in the pipeline can handle it.
      }
    });
  }

private:
  bool allowCollectives = false;

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

