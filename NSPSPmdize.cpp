//===- NSPSpmdize.cpp - NSP SPMDization -----------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Implementation of the NSP SPMDization/materialization pass.
//
// This pass is intentionally a no-op for now. It exists only to satisfy
// registration/linking and to provide a placeholder where future lowering
// from shard annotations to explicit slices/collectives will live.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hexagon {
namespace {

struct NSPSpmdizePass : public PassWrapper<NSPSpmdizePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPSpmdizePass)

  StringRef getArgument() const final { return "nsp-spmdize"; }
  StringRef getDescription() const final {
    return "NSP stub SPMDization pass (no-op placeholder)";
  }

  void runOnOperation() final {
    // Intentionally empty: this is a stub pass.
    // Future implementation will:
    //  - materialize local tiles (extract_slice/subview),
    //  - insert/resolve collectives,
    //  - rewrite compute ops to operate on sharded tiles.
  }
};

} // namespace

std::unique_ptr<Pass> createNSPSpmdizePass() {
  return std::make_unique<NSPSpmdizePass>();
}

} // namespace hexagon
} // namespace mlir

