//===- NSPShardInterface.cpp - NSP Shard Interfcae Models -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Register shard::ShardingInterface external models for a small set of
// "boundary/view/sink" operations that appear in the NSP pipeline.
//
// Ops covered so far:
//   * memref.reinterpret_cast
//   * bufferization.to_tensor
//   * bufferization.materialize_in_destination
//
// Rationale
// ---------
// shard-propagation depends on shard::ShardingInterface. When an op doesn't
// implement it, propagation can stop at that boundary. These ops are common
// around bufferization boundaries and view-like transformations.
//
// Design
// ------
// * memref.reinterpret_cast:
//     View-only. If we are given a sharding for the source, propagate it.
//     Otherwise, return "empty" (cannot infer).
// * bufferization.to_tensor:
//     Memref->tensor boundary. If the caller already proposes a result sharding,
//     accept it; otherwise, return "empty".
// * bufferization.materialize_in_destination:
//     Tensor->memref sink. If the caller proposes a sharding for the tensor
//     operand, accept it; otherwise, return "empty".
//
// For these non-structured ops we implement sharding inference/annotation
// directly (instead of relying on loop iterator/indexing-map machinery).
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"

// Dialects / ops we model.
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

// Shard interface.
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"

using namespace mlir;

namespace {

/// Returns a ranked ShapedType rank from the first ranked shaped operand/result.
/// Falls back to 0 if nothing is ranked (best-effort for propagation).
static int64_t getFirstRankedShapedRank(Operation *op) {
  auto getRank = [](Type t) -> std::optional<int64_t> {
    if (auto st = dyn_cast<ShapedType>(t)) {
      if (st.hasRank())
        return st.getRank();
    }
    return std::nullopt;
  };

  for (Type t : op->getOperandTypes())
    if (auto r = getRank(t))
      return *r;
  for (Type t : op->getResultTypes())
    if (auto r = getRank(t))
      return *r;

  return 0;
}

static SmallVector<AffineMap> makeIdentityMapsForOp(Operation *op,
                                                    int64_t rank) {
  MLIRContext *ctx = op->getContext();
  AffineMap id = AffineMap::getMultiDimIdentityMap(rank, ctx);

  SmallVector<AffineMap> maps;
  maps.reserve(op->getNumOperands() + op->getNumResults());
  for (unsigned i = 0, e = op->getNumOperands() + op->getNumResults(); i < e; ++i)
    maps.push_back(id);
  return maps;
}

static SmallVector<utils::IteratorType> makeParallelIters(int64_t rank) {
  SmallVector<utils::IteratorType> iters;
  iters.reserve(rank);
  for (int64_t i = 0; i < rank; ++i)
    iters.push_back(utils::IteratorType::parallel);
  return iters;
}

/// Convert a Sharding's split-axes into a ShardingArray.
/// The interface-level ShardingOption stores loop->grid-axis assignment as
/// `SmallVector<SmallVector<GridAxis>>`. For these ops we treat each tensor
/// dimension as an "independent loop iterator", so the mapping is 1:1.
static shard::ShardingArray toShardingArray(shard::Sharding sharding) {
  shard::ShardingArray arr;
  if (!sharding)
    return arr;

  arr.reserve(sharding.getSplitAxes().size());
  for (shard::GridAxesAttr axesAttr : sharding.getSplitAxes()) {
    SmallVector<shard::GridAxis> axes;
    auto ref = axesAttr.asArrayRef();
    axes.reserve(ref.size());
    for (int16_t a : ref)
      axes.push_back(a);
    arr.push_back(std::move(axes));
  }

  // Keep at least one sub-array to distinguish "not sharded" from "no info".
  shard::removeTrailingEmptySubArray(arr);
  return arr;
}

/// Build a Sharding from a ShardingOption by treating each entry of the
/// shardingArray as the split-axes for the corresponding tensor dimension.
static shard::Sharding fromShardingOption(Operation *op,
                                          const shard::ShardingOption &opt,
                                          int64_t rank) {
  if (!opt.grid)
    return shard::Sharding();

  SmallVector<shard::GridAxesAttr> splitAxes;
  splitAxes.reserve(rank);

  MLIRContext *ctx = op->getContext();
  ArrayRef<SmallVector<shard::GridAxis>> arr = opt.shardingArray;

  for (int64_t i = 0; i < rank; ++i) {
    SmallVector<int16_t> axesI16;
    if (i < (int64_t)arr.size()) {
      axesI16.reserve(arr[i].size());
      for (shard::GridAxis a : arr[i])
        axesI16.push_back(static_cast<int16_t>(a));
    }
    splitAxes.push_back(shard::GridAxesAttr::get(ctx, axesI16));
  }

  return shard::Sharding::get(opt.grid, splitAxes);
}

/// Create a ShardingOption corresponding to a specific value sharding.
/// For these ops we encode the value sharding into shardingArray and carry the
/// grid symbol.
static FailureOr<shard::ShardingOption> makeValueShardingOption(shard::Sharding s) {
  if (!s)
    return shard::ShardingOption::makeEmpty();
  return shard::ShardingOption(toShardingArray(s), s.getGridAttr());
}

/// Partition helper: clone `op` with `partitionedOperands` and map results.
static LogicalResult partitionByCloning(Operation *op,
                                        ArrayRef<Value> partitionedOperands,
                                        IRMapping &partitionMap,
                                        OpBuilder &builder) {
  if (partitionedOperands.size() != op->getNumOperands())
    return failure();

  IRMapping localMap;
  for (auto [oldV, newV] : llvm::zip(op->getOperands(), partitionedOperands))
    localMap.map(oldV, newV);

  Operation *cloned = builder.clone(*op, localMap);
  for (auto [oldR, newR] : llvm::zip(op->getResults(), cloned->getResults()))
    partitionMap.map(oldR, newR);

  return success();
}

//===----------------------------------------------------------------------===//
// memref.reinterpret_cast
//===----------------------------------------------------------------------===//

struct ReinterpretCastShardingModel
    : public shard::ShardingInterface::ExternalModel<
          ReinterpretCastShardingModel, memref::ReinterpretCastOp> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    (void)resultShardings;

    // Expect 1 operand and 1 result.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    // Pass-through only if a sharding is provided for the operand.
    if (operandShardings.size() < 1 || !operandShardings[0])
      return shard::ShardingOption::makeEmpty();

    return makeValueShardingOption(operandShardings[0]);
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    // We cannot (and should not) insert shard.shard on memrefs.
    // Still, we expose inferred shardings to keep propagation consistent.
    auto srcTy = dyn_cast<MemRefType>(op->getOperand(0).getType());
    auto dstTy = dyn_cast<MemRefType>(op->getResult(0).getType());

    // For view-only ops, interpret shardingArray as dim-wise split axes of the
    // "logical tensor" shape (if any). If rank is unknown, just return grid-only.
    int64_t rank = 0;
    if (dstTy)
      rank = dstTy.getRank();

    shard::Sharding s = fromShardingOption(op, opt, rank);
    std::vector<shard::Sharding> res;
    res.reserve(op->getNumOperands() + op->getNumResults());

    // Operand 0 sharding
    res.push_back(s);
    // Result 0 sharding
    res.push_back(s);

    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    // No-op for memref ops.
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding> operandShardings,
                          ArrayRef<shard::Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTableCollection,
                          OpBuilder &builder) const {
    (void)operandShardings;
    (void)resultShardings;
    (void)symbolTableCollection;
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// bufferization.to_tensor
//===----------------------------------------------------------------------===//

struct ToTensorShardingModel
    : public shard::ShardingInterface::ExternalModel<ToTensorShardingModel,
                                                    bufferization::ToTensorOp> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    // Expect 1 operand and 1 result.
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    // Prefer a caller-provided result sharding.
    if (resultShardings.size() >= 1 && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    // Fallback: if the boundary carries operand sharding information (e.g. the
    // memref was derived from a previously-sharded tensor), preserve it.
    // This avoids losing sharding across memref<->tensor conversions in some
    // pipelines.
    if (operandShardings.size() >= 1 && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    // Operand is memref (we don't annotate). Result is ranked tensor.
    auto resTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    int64_t rank = resTy ? resTy.getRank() : 0;

    shard::Sharding resultSharding = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(op->getNumOperands() + op->getNumResults());
    res.push_back(shard::Sharding());      // memref operand: no sharding
    res.push_back(resultSharding);         // tensor result
    return res;
  }

  /// Materializes sharding attaching a `shard.shard` op to the tensor result.
  /// This op is a memref→tensor boundary and cannot carry sharding on the
  /// memref operand. We therefore synthesize a `shard.sharding` value from the
  /// inferred ShardingOption and annotate only the tensor result.
  LogicalResult addShardingAnnotations(Operation *op, OpBuilder &b,
                                      const shard::ShardingOption &opt) const {

    if (op->getNumResults() != 1)
      return failure();

    Value res = op->getResult(0);
    auto resTy = dyn_cast<RankedTensorType>(res.getType());
    if (!resTy)
      return failure();

    // If the option does not prescribe a grid, there is no sharding to attach.
    if (!opt.grid)
      return success();

    const int64_t rank = resTy.getRank();
    MLIRContext *ctx = op->getContext();

    // Materialize a !shard.sharding SSA value from the ShardingOption.
    SmallVector<shard::GridAxesAttr> splitAxes;
    splitAxes.reserve(rank);

    ArrayRef<SmallVector<shard::GridAxis>> arr = opt.shardingArray;
    for (int64_t i = 0; i < rank; ++i) {
      SmallVector<int16_t> axesI16;
      if (i < (int64_t)arr.size()) {
        axesI16.reserve(arr[i].size());
        for (shard::GridAxis a : arr[i])
          axesI16.push_back(static_cast<int16_t>(a));
      }
      splitAxes.push_back(shard::GridAxesAttr::get(ctx, axesI16));
    }

    b.setInsertionPointAfter(op);

    // Build shard.sharding using the overload that takes ArrayRef<GridAxesAttr>.
    auto shardingVal =
        b.create<shard::ShardingOp>(op->getLoc(), opt.grid, splitAxes);

    // Annotate the tensor result.
    auto annotated = b.create<shard::ShardOp>(op->getLoc(), resTy, res,
                                             shardingVal.getResult());

    res.replaceAllUsesExcept(annotated.getResult(), annotated);
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding> operandShardings,
                          ArrayRef<shard::Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTableCollection,
                          OpBuilder &builder) const {
    (void)operandShardings;
    (void)resultShardings;
    (void)symbolTableCollection;
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// bufferization.materialize_in_destination
//===----------------------------------------------------------------------===//

struct MaterializeInDestShardingModel
    : public shard::ShardingInterface::ExternalModel<
          MaterializeInDestShardingModel,
          bufferization::MaterializeInDestinationOp> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    (void)resultShardings;

    // No results; we only care about the sharding of the tensor operand (usually operand 0).
    if (op->getNumOperands() < 1)
      return failure();

    if (operandShardings.size() >= 1 && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op,
                         const shard::ShardingOption &opt) const {
    // Operands: (tensor, memref). We only return sharding for the tensor operand.
    auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t rank = tensorTy ? tensorTy.getRank() : 0;

    shard::Sharding tensorSharding = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(op->getNumOperands() + op->getNumResults());

    // operand 0: tensor
    res.push_back(tensorSharding);
    // operand 1+: memrefs or other operands => no sharding
    for (unsigned i = 1; i < op->getNumOperands(); ++i)
      res.push_back(shard::Sharding());

    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    // Sink op: do not add annotations here.
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding> operandShardings,
                          ArrayRef<shard::Sharding> resultShardings,
                          IRMapping &partitionMap,
                          SymbolTableCollection &symbolTableCollection,
                          OpBuilder &builder) const {
    (void)operandShardings;
    (void)resultShardings;
    (void)symbolTableCollection;
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

} // namespace

namespace mlir {
namespace hexagon {

/// Register all external sharding interface models used by the NSP pipeline.
///
/// Call this during compiler initialization (DialectRegistry setup).
void registerNSPShardInterfaceModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    (void)dialect;
    memref::ReinterpretCastOp::attachInterface<ReinterpretCastShardingModel>(
        *ctx);
  });

  registry.addExtension(
      +[](MLIRContext *ctx, bufferization::BufferizationDialect *dialect) {
        (void)dialect;
        bufferization::ToTensorOp::attachInterface<ToTensorShardingModel>(*ctx);
        bufferization::MaterializeInDestinationOp::attachInterface<
            MaterializeInDestShardingModel>(*ctx);
      });
}

} // namespace hexagon
} // namespace mlir
