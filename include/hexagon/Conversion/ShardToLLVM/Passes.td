//===- Passes.td - Shard to LLVM passes --------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Converts shard dialect ops used by NSP materialization into standard
// arith/tensor ops so the existing linalg-to-llvm pipeline can finish lowering.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShardToLLVM : Pass<"shard-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower shard dialect elementwise ops (e.g., vadd materialized) to arith/tensor";
  let constructor = "mlir::hexagon::createShardToLLVMPass()";

  // ps. These defaults match the vadd.mlir func.func
  // linearIdx = coreId * numThreadsPerCore + threadId
  let options = [
    Option<"numThreadsPerCoreArgIndex", "ntpc-arg", "int", /*default=*/"6",
           "Function argument index for numThreadsPerCore (index-typed).">,
    Option<"threadIdArgIndex", "tid-arg", "int", /*default=*/"9",
           "Function argument index for threadId (index-typed).">,
    Option<"coreIdArgIndex", "cid-arg", "int", /*default=*/"10",
           "Function argument index for coreId (index-typed).">
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::func::FuncDialect",
    "mlir::shard::ShardDialect"
  ];
}

#endif // HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD
