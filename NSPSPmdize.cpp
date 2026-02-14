//===- NSPSpmdize.cpp - NSP SPMDization -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// Stub implementation of the NSP SPMDization/materialization pass.
//
// This pass currently performs only validation and does not rewrite IR.
// It exists to satisfy pass registration/linking and to provide a clean
// placeholder for the future materialization step.
//
// Expected IR (high level):
//   - A shard.grid symbol exists in the module (e.g. @nsp).
//   - Sharding descriptors (!shard.sharding) are produced and attached/used
//     as part of the propagation/planning flow.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

// Shard dialect ops/types.
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

namespace mlir {
namespace hexagon {

namespace {

struct NSPSpmdizePass
    : public PassWrapper<NSPSpmdizePass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPSpmdizePass)

  StringRef getArgument() const final { return "nsp-spmdize"; }

  StringRef getDescription() const final {
    return "NSP stub SPMDization pass (validation-only placeholder)";
    // return "SPMD transformation for NSP multi-core execution";
  }

  std::unique_ptr<Pass> createNSPSpmdizePass() {
    return std::make_unique<NSPSpmdizePass>();
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();
    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();

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

    // Optionally sanity-check that we have some sharding descriptors.
    // This is a warning (not a hard error) because some pipelines may
    // legally run with no sharding yet (e.g. early bring-up).
    int64_t numShardingOps = 0;
    module.walk([&](mlir::shard::ShardingOp op) { ++numShardingOps; });

    if (numShardingOps == 0) {
      module.emitWarning()
          << "NSPSpmdizePass found shard.grid '@nsp' but no 'shard.sharding' "
             "ops in the module. This pass is currently validation-only; "
             "if this is unexpected, ensure sharding propagation created "
             "sharding descriptors.";
    }

    // No-op transformation for now (future materialization will go here).
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

