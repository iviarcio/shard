//===- ShardToLLVMPass.cpp - Lower shard ops to arith/tensor ---------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Pass for lowering shard dialect ops produced by NSP sharding/materialization.
// This pass is intentionally narrow: it targets the exact shard ops used by vadd
// (shard.sharding, shard.shard, shard.process_linear_index, shard.all_slice) and
// rewrites them into standard MLIR (arith/tensor) so the existing linalg-to-llvm
// pipeline can finish the lowering.
//
//===----------------------------------------------------------------------===//

#include "hexagon/Common/Common.h"
#include "hexagon/Conversion/ShardToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "shard-to-llvm"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

#define GEN_PASS_CLASSES
#include "hexagon/Conversion/ShardToLLVM/Passes.h.inc"

namespace {

// Pick linear-index components by fixed offsets from the last function arg.
// Right-to-left (last -> first): [0], cid, tid, [1], num_cores, ntpc, ...
//
// Therefore, with last = args[N-1]:
//   cid  = args[N-2]
//   tid  = args[N-3]
//   ntpc = args[N-6]
//===----------------------------------------------------------------------===//

struct LinearIdxABI {
  Value cid;
  Value tid;
  Value ntpc;
  Value numCores; // optional
};

static Value castToIndexIfNeeded(Value v, OpBuilder &b, Location loc) {
  if (v.getType().isIndex())
    return v;
  if (isa<IntegerType>(v.getType()))
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), v);
  return Value(); // Unexpected type.
}

static FailureOr<LinearIdxABI> resolveLinearIdxABIFromTail(func::FuncOp func) {
  auto args = func.getArguments();
  int64_t n = (int64_t)args.size();

  // Need at least 6 tail slots to access ntpc at N-6.
  if (n < 6)
    return failure();

  LinearIdxABI abi;
  abi.cid     = args[n - 2];
  abi.tid     = args[n - 3];
  abi.numCores= args[n - 5]; // not required for linearIdx, but kept for future use
  abi.ntpc    = args[n - 6];

  return abi;
}

/// Compute linear process id:
///   linearIdx = coreId * numThreadsPerCore + threadId
static FailureOr<Value> computeLinearIdxFromFuncArgs(
       func::FuncOp func, OpBuilder &b, Location loc) {
  auto abiOrFail = resolveLinearIdxABIFromTail(func);
  if (failed(abiOrFail))
    return failure();

  LinearIdxABI abi = *abiOrFail;

  Value cid  = castToIndexIfNeeded(abi.cid, b, loc);
  Value tid  = castToIndexIfNeeded(abi.tid, b, loc);
  Value ntpc = castToIndexIfNeeded(abi.ntpc, b, loc);

  if (!cid || !tid || !ntpc)
    return failure();

  Value mul = b.create<arith::MulIOp>(loc, cid, ntpc);
  Value add = b.create<arith::AddIOp>(loc, mul, tid);
  return add;
}

/// Lower shard.process_linear_index to: linearIdx = cid * ntpc + tid
struct LowerProcessLinearIndex final : public RewritePattern {
  explicit LowerProcessLinearIndex(MLIRContext *ctx)
      : RewritePattern("shard.process_linear_index", /*benefit=*/1, ctx) {}
 
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    Location loc = op->getLoc();
    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, rewriter, loc);
    if (failed(linearIdxOrFail))
      return failure();

    rewriter.replaceOp(op, *linearIdxOrFail);
    return success();
  }

};

/// Lower shard.all_slice(%tensor, %sharding {grid_axes=[...], slice_axis=i})
/// into tensor.extract_slice, with offset computed from linearIdx.
/// We support only ranked 1-D tensors with static slice size.
struct LowerAllSlice final : public RewritePattern {
  explicit LowerAllSlice(MLIRContext *ctx)
      : RewritePattern("shard.all_slice", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();

    auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    auto inputTy  = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!resultTy || !inputTy)
      return failure();

    // Only 1-D tensor slicing.
    if (resultTy.getRank() != 1 || inputTy.getRank() != 1)
      return failure();

    // Only static size.
    int64_t sliceSize = resultTy.getShape()[0];
    if (sliceSize <= 0)
      return failure();

    auto func = op->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();

    Location loc = op->getLoc();

    auto linearIdxOrFail = computeLinearIdxFromFuncArgs(func, rewriter, loc);
    if (failed(linearIdxOrFail))
      return failure();

    Value linearIdx = *linearIdxOrFail;

    // offset = linearIdx * sliceSize
    Value sliceSizeVal =
        rewriter.create<arith::ConstantIndexOp>(loc, sliceSize);
    Value offset =
        rewriter.create<arith::MulIOp>(loc, linearIdx, sliceSizeVal);

    // tensor.extract_slice %in[%offset] [%sliceSize] [1]
    SmallVector<OpFoldResult> offsets{offset};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(sliceSize)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};

    Value input = op->getOperand(0);
    auto slice =
        rewriter.create<tensor::ExtractSliceOp>(loc, resultTy, input, offsets, sizes, strides);

    rewriter.replaceOp(op, slice.getResult());
    return success();
  }

};

/// shard.shard is treated as an annotation carrier. Replace it by
/// its tensor operand (operand 0), ignoring the sharding metadata.
struct LowerShardOp final : public RewritePattern {
  explicit LowerShardOp(MLIRContext *ctx)
      : RewritePattern("shard.shard", /*benefit=*/1, ctx) {}


  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // shard.shard %tensor to %sharding ...
    // Some pipelines may also produce shard.shard with extra operands/attrs.
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return failure();

    rewriter.replaceOp(op, op->getOperand(0));
    return success();
  }
};

/// Legalize shard.sharding even if it still has uses by proactively rewriting
/// its known users (shard.shard) and then erasing the sharding op.
///
/// This avoids a common dialect-conversion pitfall where the driver visits the
/// producer (shard.sharding) before rewriting its consumers.
struct LegalizeSharding final : public RewritePattern {
  explicit LegalizeSharding(MLIRContext *ctx)
      : RewritePattern("shard.sharding", /*benefit=*/10, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    Value sh = op->getResult(0);

    // Collect users first to avoid iterator invalidation while rewriting.
    SmallVector<Operation *> users(sh.getUsers().begin(), sh.getUsers().end());

    for (Operation *u : users) {
      // In this PoC, sharding is only expected to be consumed by shard.shard.
      if (u->getName().getStringRef() != "shard.shard")
        return failure();

      if (u->getNumOperands() < 1 || u->getNumResults() != 1)
        return failure();

      // Replace shard.shard by forwarding its tensor operand (operand 0).
      rewriter.replaceOp(u, u->getOperand(0));
    }

    // After rewriting users, the sharding value should become dead.
    if (!sh.use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// shard.sharding produces a !shard.sharding value used only by shard.shard/all_slice.
/// After lowering, it should become dead. Erase if unused.
struct EraseDeadShardingValue final : public RewritePattern {
  explicit EraseDeadShardingValue(MLIRContext *ctx)
      : RewritePattern("shard.sharding", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    if (!op->getResult(0).use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

/// shard.grid is metadata for the grid configuration; erase it once unused.
struct EraseGrid final : public RewritePattern {
  explicit EraseGrid(MLIRContext *ctx)
      : RewritePattern("shard.grid", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Be conservative if it unexpectedly has results that are still used.
    for (Value r : op->getResults())
      if (!r.use_empty())
        return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct ShardToLLVMPass : public ShardToLLVMBase<ShardToLLVMPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    func::FuncDialect, shard::ShardDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    ConversionTarget target(*ctx);
    target.addIllegalDialect<shard::ShardDialect>();
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect, func::FuncDialect>();

    // Everything is legal unless it belongs to the 'shard' namespace.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() != StringRef("shard");
    });

    RewritePatternSet patterns(ctx);
    patterns.add<LegalizeSharding>(ctx); 
    patterns.add<LowerShardOp>(ctx);
    patterns.add<LowerProcessLinearIndex>(ctx);
    patterns.add<LowerAllSlice>(ctx);
    patterns.add<EraseDeadShardingValue>(ctx);
    patterns.add<EraseGrid>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hexagon::createShardToLLVMPass() {
  return std::make_unique<ShardToLLVMPass>();
}
