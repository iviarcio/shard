//===- NSPMaterializePass.cpp - NSP destination materialization -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP materialization pass.
//
// This pass performs the destination-materialization half of the split NSP
// bring-up pipeline for multi-core execution. It expects that a prior
// NSPLocalizePass has:
//   - cloned supported sharded elementwise computations into shard-local
//     tensor semantics,
//   - left the original global tensor producer intact,
//   - attached the following temporary attributes to the shard-local generic:
//       * nsp.localized
//       * nsp.materialize_tile_size
//       * nsp.materialize_split_axis
//       * nsp.group_id
//   - attached nsp.group_id to the original global generic as well.
//
// For each localized/global pair, this pass:
//   1. re-discovers the original materialize_in_destination sink from the
//      global generic result,
//   2. computes the participant-local tile offset using
//      shard.process_linear_index,
//   3. creates a memref.subview into the destination buffer,
//   4. rewrites the existing materialization sink to store the shard-local
//      tensor into that subview,
//   5. erases the trivial wrapper chain and the original global generic.
//
// Bring-up constraints:
//   - only handles localized linalg.generic ops produced by NSPLocalizePass,
//   - only handles destinations reachable through a single-use wrapper chain
//     consisting of shard.shard and optional tensor.cast,
//   - only handles rank-1 direct store-by-tile materialization.
//
// Expected pipeline shape:
//   planner -> propagation -> nsp-localize -> tiling/vectorization
//           -> nsp-materialize -> cleanup/lowering
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

namespace mlir {
namespace hexagon {

namespace {

// Find a bufferization.materialize_in_destination sink for `v` while allowing
// a trivial chain of sharding wrappers.
//
// This mirrors the discovery logic used by the original monolithic
// NSPSpmdize pass so the split localize/materialize pipeline preserves the
// previous sink-finding behavior.
static std::optional<std::pair<bufferization::MaterializeInDestinationOp,
                               SmallVector<Operation *>>>
findMaterializeSink(Value v) {
  SmallVector<Operation *> wrappers;
  Value cur = v;

  while (true) {
    if (!cur.hasOneUse())
      return std::nullopt;

    Operation *user = *cur.getUsers().begin();

    if (auto shardOp = dyn_cast<mlir::shard::ShardOp>(user)) {
      if (shardOp->getNumOperands() < 1 || shardOp->getOperand(0) != cur)
        return std::nullopt;
      wrappers.push_back(user);
      cur = shardOp->getResult(0);
      continue;
    }

    if (auto castOp = dyn_cast<tensor::CastOp>(user)) {
      if (castOp.getSource() != cur)
        return std::nullopt;
      wrappers.push_back(user);
      cur = castOp.getResult();
      continue;
    }

    if (auto mat = dyn_cast<bufferization::MaterializeInDestinationOp>(user)) {
      if (mat->getOperand(0) != cur)
        return std::nullopt;
      return std::make_optional(std::make_pair(mat, wrappers));
    }

    return std::nullopt;
  }
}

// Rewrite the existing sink to materialize a shard-local tensor tile into a
// subview of the final destination buffer.
static LogicalResult
materializeTileToDestination(OpBuilder &b, linalg::GenericOp localizedGeneric,
                             bufferization::MaterializeInDestinationOp mat,
                             ArrayRef<Operation *> wrappers,
                             linalg::GenericOp originalGeneric, Value procIdx) {

  if (!localizedGeneric->hasAttr("nsp.localized"))
    return failure();

  auto tileSizeAttr =
      localizedGeneric->getAttrOfType<IntegerAttr>("nsp.materialize_tile_size");
  auto splitAxisAttr = localizedGeneric->getAttrOfType<IntegerAttr>(
      "nsp.materialize_split_axis");
  if (!tileSizeAttr || !splitAxisAttr)
    return failure();

  Location loc = localizedGeneric.getLoc();
  int64_t tileSize = tileSizeAttr.getInt();
  int64_t splitAxis = splitAxisAttr.getInt();

  Value dest = mat.getOperand(1);
  auto destTy = dyn_cast<MemRefType>(dest.getType());
  if (!destTy)
    return failure();

  if (destTy.getRank() != 1)
    return failure();

  if (splitAxis < 0 || splitAxis >= destTy.getRank())
    return failure();

  b.setInsertionPoint(mat);

  Value tileSizeVal = b.create<arith::ConstantIndexOp>(loc, tileSize);
  Value offset = b.create<arith::MulIOp>(loc, procIdx, tileSizeVal);

  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.reserve(destTy.getRank());
  sizes.reserve(destTy.getRank());
  strides.reserve(destTy.getRank());

  for (int64_t i = 0; i < destTy.getRank(); ++i) {
    strides.push_back(b.getIndexAttr(1));
    if (i == splitAxis) {
      offsets.push_back(offset);
      sizes.push_back(b.getIndexAttr(tileSize));
    } else {
      offsets.push_back(b.getIndexAttr(0));
      int64_t dim = destTy.getDimSize(i);
      if (ShapedType::isDynamic(dim))
        return failure();
      sizes.push_back(b.getIndexAttr(dim));
    }
  }

  auto subLayout = StridedLayoutAttr::get(destTy.getContext(),
                                          /*offset=*/ShapedType::kDynamic,
                                          /*strides=*/ArrayRef<int64_t>{1});
  auto subviewTy =
      MemRefType::get(ArrayRef<int64_t>{tileSize}, destTy.getElementType(),
                      subLayout, destTy.getMemorySpace());

  Value destSubview = b.create<memref::SubViewOp>(loc, subviewTy, dest, offsets,
                                                  sizes, strides);

  // Reuse the pre-existing sink instead of creating a second
  // materialize_in_destination op.
  mat->setOperand(0, localizedGeneric.getResult(0));
  mat->setOperand(1, destSubview);

  for (Operation *op : llvm::reverse(wrappers))
    op->erase();

  originalGeneric.erase();

  localizedGeneric->removeAttr("nsp.localized");
  localizedGeneric->removeAttr("nsp.materialize_tile_size");
  localizedGeneric->removeAttr("nsp.materialize_split_axis");
  localizedGeneric->removeAttr("nsp.group_id");

  return success();
}

struct NSPMaterializePass
    : public PassWrapper<NSPMaterializePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPMaterializePass)

  StringRef getArgument() const final { return "nsp-materialize"; }

  StringRef getDescription() const final {
    return "Materialize shard-local tensor results into destination memrefs";
  }

  void runOnOperation() final {
    MLIRContext *ctx = &getContext();

    ctx->getOrLoadDialect<mlir::shard::ShardDialect>();
    ctx->getOrLoadDialect<mlir::arith::ArithDialect>();
    ctx->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    ctx->getOrLoadDialect<mlir::func::FuncDialect>();
    ctx->getOrLoadDialect<mlir::linalg::LinalgDialect>();
    ctx->getOrLoadDialect<mlir::memref::MemRefDialect>();
    ctx->getOrLoadDialect<mlir::tensor::TensorDialect>();

    ModuleOp module = getOperation();

    auto grid = module.lookupSymbol<mlir::shard::GridOp>("nsp");
    if (!grid) {
      module.emitError()
          << "NSPMaterializePass expected a 'shard.grid' symbol named '@nsp' "
             "in the module, but none was found. "
             "Ensure NSP shard planning/localization ran and created the grid.";
      signalPassFailure();
      return;
    }

    module.walk([&](mlir::func::FuncOp func) {
      OpBuilder b(func.getContext());

      SmallVector<linalg::GenericOp> localizedWorklist;
      func.walk([&](linalg::GenericOp g) {
        if (g->hasAttr("nsp.localized"))
          localizedWorklist.push_back(g);
      });

      DenseMap<int64_t, linalg::GenericOp> originalByGroupId;
      func.walk([&](linalg::GenericOp g) {
        if (g->hasAttr("nsp.localized"))
          return;
        if (auto id = g->getAttrOfType<IntegerAttr>("nsp.group_id"))
          originalByGroupId[id.getInt()] = g;
      });

      for (linalg::GenericOp localizedGeneric : localizedWorklist) {
        auto groupIdAttr =
            localizedGeneric->getAttrOfType<IntegerAttr>("nsp.group_id");
        if (!groupIdAttr) {
          localizedGeneric.emitError()
              << "NSPMaterialize: localized op is missing required "
                 "'nsp.group_id' attribute";
          signalPassFailure();
          return;
        }

        auto it = originalByGroupId.find(groupIdAttr.getInt());
        if (it == originalByGroupId.end()) {
          localizedGeneric.emitError()
              << "NSPMaterialize: could not find matching original global "
                 "generic for group id "
              << groupIdAttr.getInt();
          signalPassFailure();
          return;
        }

        linalg::GenericOp originalGeneric = it->second;
        if (!originalGeneric)
          continue;

        auto sink = findMaterializeSink(originalGeneric.getResult(0));
        if (!sink) {
          originalGeneric.emitError()
              << "NSPMaterialize: failed to re-discover "
                 "materialize_in_destination sink for group id "
              << groupIdAttr.getInt();
          signalPassFailure();
          return;
        }

        auto mat = sink->first;
        auto &wrappers = sink->second;

        Location loc = localizedGeneric.getLoc();
        b.setInsertionPoint(mat);
        Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);

        if (failed(materializeTileToDestination(b, localizedGeneric, mat,
                                                wrappers, originalGeneric,
                                                procIdx))) {
          localizedGeneric.emitError()
              << "NSPMaterialize: failed to materialize local tile into "
                 "destination";
          signalPassFailure();
          return;
        }

      } // for: localizedWorklist
    }); // module walk
  } // runOnOperation
};

} // namespace

std::unique_ptr<Pass> createNSPMaterializePass() {
  return std::make_unique<NSPMaterializePass>();
}

} // namespace hexagon
} // namespace mlir
