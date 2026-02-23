//===- Passes.td - Shard to LLVM passes --------------------*- tablegen -*-===//
//
// Converts shard dialect ops used by NSP materialization into standard
// arith/tensor ops so the existing linalg-to-llvm pipeline can finish lowering.
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD
#define HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ShardToLLVM : Pass<"shard-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower shard dialect ops (PoC: vadd materialization) to arith/tensor";
  let constructor = "mlir::hexagon::createShardToLLVMPass()";

  // These defaults match the vadd.mlir PoC
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
