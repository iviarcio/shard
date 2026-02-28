//===- NSPShardInterface.cpp - NSP Shard Interface Models -----------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Register shard::ShardingInterface external models for a focused set of
// "boundary / view / sink / structural" operations that appear in the NSP
// sharding + SPMD pipeline.
//
// Rationale
// ---------
// shard-propagation depends on shard::ShardingInterface. If an operation does
// not implement it, propagation may stop at that boundary.
//
// This file covers non-structured ops that commonly appear around:
//   * View-like transformations
//   * Allocation / copy boundaries
//   * Bufferization boundaries (tensor <-> memref)
//   * Scalar / control-flow constructs
//   * Elementwise math/arith ops (e.g., math.exp, math.tanh)
//
// Design Principles
// -----------------
// * Preserve existing sharding when semantically safe.
// * Treat boundaries as propagation boundaries, not compute semantics.
// * For elementwise ops, propagate sharding operand <-> result.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"

// Dialects / ops we model.
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

// Shard interface.
#include "mlir/Dialect/Shard/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Shard/Interfaces/ShardingInterfaceImpl.h"

using namespace mlir;

namespace {

/// Returns a ranked shaped rank from the first ranked shaped operand/result.
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

static bool hasRankedShapedType(Operation *op) {
  auto isRankedShaped = [](Type t) {
    auto st = dyn_cast<ShapedType>(t);
    return st && st.hasRank();
  };
  for (Type t : op->getOperandTypes())
    if (isRankedShaped(t))
      return true;
  for (Type t : op->getResultTypes())
    if (isRankedShaped(t))
      return true;
  return false;
}

static SmallVector<AffineMap> makeIdentityMapsForOp(Operation *op, int64_t rank) {
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
/// For these ops we treat each tensor dimension as an independent iterator.
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

/// Build a Sharding from a ShardingOption by treating each shardingArray entry
/// as the split-axes for the corresponding tensor dimension.
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
// Generic models
//===----------------------------------------------------------------------===//

/// Generic model for scalar/control-flow ops that should not participate in sharding.
/// It tells shard-propagation to ignore the op instead of erroring out.
template <typename OpTy>
struct NoShardingModel
    : public shard::ShardingInterface::ExternalModel<NoShardingModel<OpTy>, OpTy> {
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *) const { return {}; }
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }
  SmallVector<AffineMap> getIndexingMaps(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *, ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>) const {
    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

/// Generic elementwise model for unary/binary arith/math ops.
/// For ranked shaped types, propagate sharding operand <-> result.
/// For pure scalars, behave like NoShardingModel (empty info).
template <typename OpTy>
struct ElementwiseShardingModel
    : public shard::ShardingInterface::ExternalModel<ElementwiseShardingModel<OpTy>, OpTy> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    // Scalar-only instances: no sharding info.
    if (!hasRankedShapedType(op))
      return shard::ShardingOption::makeEmpty();

    // Prefer the first known operand sharding.
    for (shard::Sharding s : operandShardings)
      if (s)
        return makeValueShardingOption(s);

    // Otherwise accept a proposed result sharding.
    for (shard::Sharding s : resultShardings)
      if (s)
        return makeValueShardingOption(s);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    if (!hasRankedShapedType(op))
      return res;

    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    for (auto &x : res)
      x = s;
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    // Elementwise ops do not need explicit shard ops inserted.
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

/// Result-only model for ops with 0 operands and (typically) 1 result.
/// Useful for tensor constants: accept a proposed result sharding when present.
template <typename OpTy>
struct ResultOnlyShardingModel
    : public shard::ShardingInterface::ExternalModel<ResultOnlyShardingModel<OpTy>, OpTy> {

  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }

  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    if (!hasRankedShapedType(op))
      return {};
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }

  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const {
    return {};
  }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (!hasRankedShapedType(op))
      return shard::ShardingOption::makeEmpty();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());

    if (!hasRankedShapedType(op))
      return res;

    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);
    for (auto &x : res)
      x = s;
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// memref.reinterpret_cast / memref.alloc / memref.copy
//===----------------------------------------------------------------------===//

struct ReinterpretCastShardingModel
    : public shard::ShardingInterface::ExternalModel<ReinterpretCastShardingModel,
                                                    memref::ReinterpretCastOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding> resultShardings) const {
    (void)resultShardings;
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    if (!operandShardings.empty() && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct AllocShardingModel
    : public shard::ShardingInterface::ExternalModel<AllocShardingModel, memref::AllocOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (op->getNumOperands() != 0 || op->getNumResults() != 1)
      return failure();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(1);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct CopyShardingModel
    : public shard::ShardingInterface::ExternalModel<CopyShardingModel, memref::CopyOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding>) const {
    if (op->getNumOperands() != 2 || op->getNumResults() != 0)
      return failure();

    // Prefer src sharding; otherwise accept dst sharding.
    if (operandShardings.size() >= 1 && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);
    if (operandShardings.size() >= 2 && operandShardings[1])
      return makeValueShardingOption(operandShardings[1]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s); // src
    res.push_back(s); // dst
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

//===----------------------------------------------------------------------===//
// bufferization.to_tensor / bufferization.materialize_in_destination
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
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding>,
                    ArrayRef<shard::Sharding> resultShardings) const {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();

    if (!resultShardings.empty() && resultShardings[0])
      return makeValueShardingOption(resultShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.reserve(2);
    res.push_back(s);
    res.push_back(s);
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

struct MaterializeInDestShardingModel
    : public shard::ShardingInterface::ExternalModel<MaterializeInDestShardingModel,
                                                    bufferization::MaterializeInDestinationOp> {
  SmallVector<AffineMap> getIndexingMaps(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeIdentityMapsForOp(op, rank);
  }
  SmallVector<utils::IteratorType> getLoopIteratorTypes(Operation *op) const {
    int64_t rank = getFirstRankedShapedRank(op);
    return makeParallelIters(rank);
  }
  SmallVector<shard::ReductionKind> getReductionLoopIteratorKinds(Operation *) const { return {}; }

  FailureOr<shard::ShardingOption>
  getShardingOption(Operation *op, ArrayRef<shard::Sharding> operandShardings,
                    ArrayRef<shard::Sharding>) const {
    if (op->getNumOperands() != 2 || op->getNumResults() != 0)
      return failure();

    // Operand(0) is the tensor to materialize. If it has sharding, accept it.
    if (!operandShardings.empty() && operandShardings[0])
      return makeValueShardingOption(operandShardings[0]);

    return shard::ShardingOption::makeEmpty();
  }

  FailureOr<std::vector<shard::Sharding>>
  getShardingAnnotations(Operation *op, const shard::ShardingOption &opt) const {
    int64_t rank = getFirstRankedShapedRank(op);
    shard::Sharding s = fromShardingOption(op, opt, rank);

    std::vector<shard::Sharding> res;
    res.resize(op->getNumOperands() + op->getNumResults(), shard::Sharding());
    // Both operands get the same logical sharding info (tensor + destination memref).
    if (res.size() >= 2) {
      res[0] = s;
      res[1] = s;
    }
    return res;
  }

  LogicalResult addShardingAnnotations(Operation *, OpBuilder &,
                                      const shard::ShardingOption &) const {
    return success();
  }

  LogicalResult partition(Operation *op, ArrayRef<Value> partitionedOperands,
                          ArrayRef<shard::Sharding>, ArrayRef<shard::Sharding>,
                          IRMapping &partitionMap, SymbolTableCollection &,
                          OpBuilder &builder) const {
    return partitionByCloning(op, partitionedOperands, partitionMap, builder);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexagon {

/// Register all external sharding interface models used by the NSP pipeline.
/// This must be called during compiler initialization (DialectRegistry setup).
void registerNSPShardInterfaceModels(DialectRegistry &registry) {
  // Memref boundaries / structural ops.
  registry.addExtension(+[](MLIRContext *ctx, memref::MemRefDialect *dialect) {
    (void)dialect;
    memref::ReinterpretCastOp::attachInterface<ReinterpretCastShardingModel>(*ctx);
    memref::AllocOp::attachInterface<AllocShardingModel>(*ctx);
    memref::CopyOp::attachInterface<CopyShardingModel>(*ctx);
  });

  // Bufferization boundaries.
  registry.addExtension(+[](MLIRContext *ctx, bufferization::BufferizationDialect *dialect) {
    (void)dialect;
    bufferization::ToTensorOp::attachInterface<ToTensorShardingModel>(*ctx);
    bufferization::MaterializeInDestinationOp::attachInterface<
        MaterializeInDestShardingModel>(*ctx);
  });

  // Control-flow (modeled as sharding-transparent for now).
  registry.addExtension(+[](MLIRContext *ctx, scf::SCFDialect *dialect) {
    (void)dialect;
    scf::ForOp::attachInterface<NoShardingModel<scf::ForOp>>(*ctx);
    scf::YieldOp::attachInterface<NoShardingModel<scf::YieldOp>>(*ctx);
  });

  // Arith: mix of scalar and tensor elementwise ops.
  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    (void)dialect;

    // Constants may produce tensors; accept proposed result sharding.
    arith::ConstantOp::attachInterface<ResultOnlyShardingModel<arith::ConstantOp>>(*ctx);

    // Casts / loop-bound plumbing ops (scalar/control oriented).
    arith::IndexCastOp::attachInterface<NoShardingModel<arith::IndexCastOp>>(*ctx);

    // Elementwise (works for tensors; becomes "empty" for scalars).
    arith::AddFOp::attachInterface<ElementwiseShardingModel<arith::AddFOp>>(*ctx);
    arith::SubFOp::attachInterface<ElementwiseShardingModel<arith::SubFOp>>(*ctx);
    arith::MulFOp::attachInterface<ElementwiseShardingModel<arith::MulFOp>>(*ctx);
    arith::DivFOp::attachInterface<ElementwiseShardingModel<arith::DivFOp>>(*ctx);

    arith::AddIOp::attachInterface<ElementwiseShardingModel<arith::AddIOp>>(*ctx);
    arith::SubIOp::attachInterface<ElementwiseShardingModel<arith::SubIOp>>(*ctx);
    arith::MulIOp::attachInterface<ElementwiseShardingModel<arith::MulIOp>>(*ctx);

    // Add more as they appear.
  });

  // Math: Commonly used ops: exp/tanh/erf
  registry.addExtension(+[](MLIRContext *ctx, math::MathDialect *dialect) {
    (void)dialect;
    math::ExpOp::attachInterface<ElementwiseShardingModel<math::ExpOp>>(*ctx);
    math::TanhOp::attachInterface<ElementwiseShardingModel<math::TanhOp>>(*ctx);
    math::ErfOp::attachInterface<ElementwiseShardingModel<math::ErfOp>>(*ctx);

    // Add more as they appear (e.g., math::RsqrtOp, math::LogOp, ...).
  });
}

} // namespace hexagon
} // namespace mlir
