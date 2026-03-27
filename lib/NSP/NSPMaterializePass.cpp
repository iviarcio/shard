//===- NSPMaterializePass.cpp - NSP destination materialization -----------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
//
//===----------------------------------------------------------------------===//
//
// NSP materialization pass.
//
// This pass consumes the explicit `nsp.materialize_tile` hand-off operation
// emitted by NSPLocalizePass in non-collective mode.
//
// For each `nsp.materialize_tile %tile into %dest grid @g ...`, the pass:
//   1. computes the participant-local tile offset using
//      shard.process_linear_index,
//   2. creates a memref.subview into the final destination buffer,
//   3. materializes the shard-local tensor tile into that subview using
//      bufferization.materialize_in_destination,
//   4. erases the temporary NSP hand-off op.
//
// Bring-up constraints:
//   - only handles rank-1 direct store-by-tile materialization,
//   - expects static tile size,
//   - expects a valid shard.grid symbol referenced by the NSP op.
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
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"

#include "hexagon/NSP/IR/NSPDialect.h"
#include "hexagon/NSP/IR/NSPOps.h"

namespace mlir {
namespace hexagon {

namespace {

static FailureOr<mlir::shard::GridOp>
lookupGrid(Operation *op, FlatSymbolRefAttr gridAttr) {
  if (!gridAttr)
    return failure();

  auto grid = SymbolTable::lookupNearestSymbolFrom<mlir::shard::GridOp>(op,
                                                                        gridAttr);
  if (!grid)
    return failure();

  return grid;
}

static LogicalResult materializeTileToDestination(
    OpBuilder &b, mlir::hexagon::nsp::MaterializeTileOp tileOp,
    mlir::shard::GridOp grid) {
  Location loc = tileOp.getLoc();

  Value source = tileOp.getSource();
  Value dest = tileOp.getDest();

  auto sourceTy = dyn_cast<RankedTensorType>(source.getType());
  auto destTy = dyn_cast<MemRefType>(dest.getType());
  if (!sourceTy || !destTy)
    return failure();

  // Keep bring-up constraints aligned with the current NSPLocalizePass.
  if (sourceTy.getRank() != 1 || destTy.getRank() != 1)
    return failure();

  int64_t splitAxis = tileOp.getSplitAxis();
  int64_t tileSize = tileOp.getTileSize();
  if (splitAxis != 0 || tileSize <= 0)
    return failure();

  b.setInsertionPoint(tileOp);

  Value procIdx = b.create<mlir::shard::ProcessLinearIndexOp>(loc, grid);
  Value tileSizeVal = b.create<arith::ConstantIndexOp>(loc, tileSize);
  Value offset = b.create<arith::MulIOp>(loc, procIdx, tileSizeVal);

  SmallVector<OpFoldResult> offsets = {offset};
  SmallVector<OpFoldResult> sizes = {b.getIndexAttr(tileSize)};
  SmallVector<OpFoldResult> strides = {b.getIndexAttr(1)};

  auto subLayout = StridedLayoutAttr::get(
      destTy.getContext(), /*offset=*/ShapedType::kDynamic,
      /*strides=*/ArrayRef<int64_t>{1});
  auto subviewTy =
      MemRefType::get(ArrayRef<int64_t>{tileSize}, destTy.getElementType(),
                      subLayout, destTy.getMemorySpace());

  Value destSubview = b.create<memref::SubViewOp>(loc, subviewTy, dest, offsets,
                                                  sizes, strides);

  b.create<bufferization::MaterializeInDestinationOp>(loc, source, destSubview);

  if (auto localizedGeneric = source.getDefiningOp<linalg::GenericOp>()) {
    localizedGeneric->removeAttr("nsp.localized");
    localizedGeneric->removeAttr("nsp.materialize_tile_size");
    localizedGeneric->removeAttr("nsp.materialize_split_axis");
    localizedGeneric->removeAttr("nsp.group_id");
  }

  tileOp.erase();
  return success();
}

struct NSPMaterializePass
    : public PassWrapper<NSPMaterializePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NSPMaterializePass)

  StringRef getArgument() const final { return "nsp-materialize"; }

  StringRef getDescription() const final {
    return "Materialize shard-local tensor results into destination memrefs";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::shard::ShardDialect, mlir::arith::ArithDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect, mlir::tensor::TensorDialect,
                    mlir::hexagon::nsp::NSPDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();

    SmallVector<mlir::hexagon::nsp::MaterializeTileOp> worklist;
    module.walk([&](mlir::hexagon::nsp::MaterializeTileOp op) {
      worklist.push_back(op);
    });

    for (mlir::hexagon::nsp::MaterializeTileOp tileOp : worklist) {
      if (!tileOp || !tileOp->getParentOp())
        continue;

      auto gridOr = lookupGrid(tileOp, tileOp.getGridAttr());
      if (failed(gridOr)) {
        tileOp.emitError()
            << "NSPMaterialize: could not resolve grid symbol "
            << tileOp.getGridAttr();
        signalPassFailure();
        return;
      }

      OpBuilder b(tileOp);
      if (failed(materializeTileToDestination(b, tileOp, *gridOr))) {
        tileOp.emitError()
            << "NSPMaterialize: failed to materialize local tile into "
               "destination";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createNSPMaterializePass() {
  return std::make_unique<NSPMaterializePass>();
}

} // namespace hexagon
} // namespace mlir
