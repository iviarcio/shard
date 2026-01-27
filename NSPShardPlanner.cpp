//===- NSPShardPlanner.cpp --------------------------------------*- C++ -*-===//
//
// This file implements a sharding planner for NSP SPMD execution.
//
// The pass inspects linalg.generic operations and attaches shard dialect
// annotations based on iterator_types and indexing_maps. It optionally
// materializes communication (e.g. all-reduce) when the chosen sharding
// implies cross-device reductions.
//
// Design goals:
//   * Work at the "linalg.generic" level (pre-tiling).
//   * Be dialect-agnostic except for using the shard dialect as the sharding IR.
//   * Keep the pipeline compatible with further lowering to per-NSP slices
//     (tensor.extract_slice/memref.subview) and runtime intrinsics.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

// shard dialect IR (GridOp, ShardOp, collectives, attrs).
#include "mlir/Dialect/Shard/IR/ShardOps.h"

using namespace mlir;

namespace {

/// A simple policy describing how we want to shard a computation.
///
/// In practice you may want this to be configurable (via pass options),
/// or drive it from a cost model (e.g. memory traffic vs compute).
struct ShardPolicy {
  /// The number of NSP devices in the grid (1-D mesh for now).
  int64_t nspCount = 16;

  /// Prefer sharding along the first parallel iterator that indexes the output.
  bool preferFirstOutputParallelDim = true;

  /// If true, allow sharding decisions that introduce collectives (e.g. all-reduce).
  /// For your initial "no inter-NSP traffic" mode, keep this false.
  bool allowCollectives = false;
};

/// A structured description of the sharding decision for a single op.
struct ShardPlan {
  /// The chosen loop iterator index to shard (e.g. i or j in matmul).
  int64_t shardIter = -1;

  /// Whether this plan requires a cross-device all-reduce to be correct.
  bool requiresAllReduce = false;

  /// The shard axis on the device grid (only 0 for 1-D grid).
  int64_t gridAxis = 0;
};

/// Pass that attaches shard annotations to linalg.generic operations.
///
/// This pass does NOT perform SPMDization itself; it only generates a
/// declarative sharding IR. Downstream passes are expected to run:
///   * -sharding-propagation (optional but recommended)
///   * an SPMDization/materialization pass that lowers shard to slices and/or comms
///
/// See the shard dialect documentation. :contentReference[oaicite:2]{index=2}
struct NSPShardPlannerPass
    : public PassWrapper<NSPShardPlannerPass, OperationPass<func::FuncOp>> {

  NSPShardPlannerPass() = default;
  NSPShardPlannerPass(const ShardPolicy &policy) : policy(policy) {}

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();

    // Ensure we have a shard.grid symbol available in the module.
    ModuleOp module = func->getParentOfType<ModuleOp>();
    shard::GridOp grid = getOrCreateGrid(module, policy.nspCount);

    // IMPORTANT: We rebuild (and erase) linalg.generic ops during annotation.
    // Erasing ops while walking can invalidate the walk. Collect first, then
    // rewrite in a stable loop.
    SmallVector<linalg::GenericOp> generics;
    func.walk([&](linalg::GenericOp op) {
     if (op.hasTensorSemantics())
       generics.push_back(op);
    });

    for (linalg::GenericOp op : generics) {
     // The op may have been erased by a previous rewrite.
     if (!op || !op->getParentOp())
       continue;

     // 1) Analyze the op and build a sharding plan.
     FailureOr<ShardPlan> plan = buildPlanForGeneric(op, policy);
     if (failed(plan))
       continue;

     // If the plan introduces collectives but they are not allowed, bail out.
     if (plan->requiresAllReduce && !policy.allowCollectives)
       continue;

     // 2) Attach shard annotations to operands using shard.shard AND rebuild
     // the linalg.generic so that it consumes the annotated values.
     linalg::GenericOp newOp = annotateGenericWithShard(op, grid, *plan);

     // 3) If needed, insert shard collectives (e.g. all-reduce) explicitly.
     // In many pipelines, a later pass does this; but we show it here
     // to make the semantics explicit.
     if (plan->requiresAllReduce)
       materializeAllReduceIfNeeded(newOp, *plan);
    }
  }

private:
  ShardPolicy policy;

  /// Get or create `shard.grid @nsp(shape = <nspCount>)` in the module.
  ///
  /// The grid is a symbol used by shard attributes and operations.
  static shard::GridOp getOrCreateGrid(ModuleOp module, int64_t nspCount) {
    MLIRContext *ctx = module.getContext();
    OpBuilder b(ctx);

    // Look up a pre-existing grid named "nsp".
    if (auto grid = module.lookupSymbol<shard::GridOp>("nsp"))
      return grid;

    b.setInsertionPointToStart(module.getBody());
    auto shapeAttr = b.getI64ArrayAttr({nspCount});
    return b.create<shard::GridOp>(module.getLoc(), "nsp", shapeAttr);
  }

  /// Build a sharding plan for a linalg.generic op.
  ///
  /// The key inputs are:
  ///   * iterator_types: parallel vs reduction
  ///   * indexing_maps: which loop dims index which operands/results
  ///
  /// Heuristic:
  ///   * Prefer a parallel iterator that appears as a direct dim in the output map.
  ///   * If sharding would shard a reduced dimension that must be aggregated,
  ///     mark requiresAllReduce=true.
  static FailureOr<ShardPlan> buildPlanForGeneric(linalg::GenericOp op,
                                                 const ShardPolicy &policy) {
    ShardPlan plan;

    // Extract iterator types.
    SmallVector<StringRef> iters;
    iters.reserve(op.getNumLoops());
    for (Attribute a : op.getIteratorTypesArray())
      iters.push_back(cast<StringAttr>(a).getValue());

    // Identify output indexing map (assume single output for simplicity; generalize as needed).
    if (op.getNumOutputs() != 1)
      return failure();
    AffineMap outMap = op.getIndexingMapsArray().back();

    // Pick a sharding iterator.
    plan.shardIter = pickShardIteratorIndex(outMap, iters, policy);
    if (plan.shardIter < 0)
      return failure();

    // Determine if this sharding requires a cross-device all-reduce.
    //
    // Rule of thumb:
    //   * If the shardIter corresponds to a reduction-only dimension of the final value,
    //     then each device computes partial results that need aggregation.
    //
    // For elementwise ops: false.
    // For matmul sharded on i or j: false (k is local reduction).
    // For global sum sharded on reduced axis: true.
    plan.requiresAllReduce = shardingIntroducesGlobalReduction(op, plan.shardIter);

    return plan;
  }

  /// Decide which iterator to shard based on output map and iterator_types.
  ///
  /// \returns the loop iterator index, or -1 if no suitable iterator exists.
  static int64_t pickShardIteratorIndex(AffineMap outMap,
                                       ArrayRef<StringRef> iteratorTypes,
                                       const ShardPolicy &policy) {
    // We want an iterator that:
    //   (1) is "parallel"
    //   (2) appears as a plain AffineDimExpr in the output map results.
    //
    // Example: outMap = (i,j,k) -> (i,j)
    // candidates: i and j (if both are parallel iterators).
    for (AffineExpr e : outMap.getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(e);
      if (!dim)
        continue;
      int64_t iterIdx = dim.getPosition();
      if (iterIdx < 0 || iterIdx >= (int64_t)iteratorTypes.size())
        continue;
      if (iteratorTypes[iterIdx] == "parallel")
        return iterIdx;
    }
    return -1;
  }

  /// Determine whether sharding on `shardIter` forces an all-reduce for correctness.
  ///
  /// NOTE: This is intentionally conservative. A production implementation
  /// should understand:
  ///   * how results are used downstream
  ///   * whether the result is expected to be replicated or sharded
  ///   * whether reductions are local vs global
  static bool shardingIntroducesGlobalReduction(linalg::GenericOp op,
                                               int64_t shardIter) {
    // Minimal heuristic:
    // If the op returns a scalar (rank-0) and any loop is sharded, it's likely
    // a global reduction -> needs all-reduce.
    if (op.getResultTypes().size() == 1) {
      if (auto rt = dyn_cast<RankedTensorType>(op.getResultTypes()[0])) {
        if (rt.getRank() == 0)
          return true;
      }
    }

    // For typical matmul-like patterns, sharding on i/j does not need all-reduce.
    // You can recognize matmul by iterator types and maps:
    //   iterators: [parallel, parallel, reduction]
    //   maps: A(i,k), B(k,j), C(i,j)
    // If shardIter is one of the parallel iters that index C, return false.
    //
    // TODO: Implement pattern recognition here.
    return false;
  }

  /// Annotate a linalg::GenericOp with shard semantics.
  ///
  /// This function materializes sharding information for each tensor operand
  /// (inputs and outputs) of a linalg::GenericOp by:
  ///
  ///  1. Creating explicit `!shard.sharding` SSA values describing how each
  ///     tensor is partitioned across a shard grid.
  ///  2. Wrapping tensor operands with `shard.shard` ops that attach those
  ///     sharding descriptors to the SSA values.
  ///  3. Rebuilding the linalg::GenericOp so that it consumes the annotated
  ///     operands, since sharding is represented as SSA values rather than
  ///     attributes.
  ///
  /// This function is invoked once per linalg::GenericOp selected by the
  /// NSP shard planner, after a sharding plan has been computed (i.e. a loop
  /// iterator to shard has already been chosen).
  ///
  /// Sharding decisions are conservative and operand-local:
  ///  - A tensor dimension is sharded iff it is indexed directly by the
  ///    loop iterator selected for sharding.
  ///  - All other dimensions are replicated.
  ///  - The current implementation assumes a 1-D shard grid (axis 0).
  ///  - Halo sizes and offsets are not inferred here.
  ///
  /// The original linalg::GenericOp is replaced by a new one that consumes
  /// the sharded operands; the region body is moved without cloning.
  ///
  /// \returns The newly created linalg::GenericOp consuming sharded operands.
  static linalg::GenericOp annotateGenericWithShard(linalg::GenericOp op,
                                                   shard::GridOp grid,
                                                   const ShardPlan &plan) {
    // ------------------------------------------------------------------
    // Obs. on Shard dialect (based on ODS from LLVM commit: 064f02dac):
    //   %sh = shard.sharding @nsp split_axes = [...] : !shard.sharding
    //   %v2 = shard.shard %v to %sh annotate_for_users : tensor<...>
    //
    // The key point is: the "sharding description" is a Value
    // (!shard.sharding). Therefore, we must build shard.sharding first,
    // and then use it as the second operand to shard.shard.
    // ------------------------------------------------------------------

    OpBuilder b(op);
    Location loc = op.getLoc();
    MLIRContext *ctx = op.getContext();

    /// Build the `split_axes` specification for a single tensor operand.
    ///
    /// Given the operand's indexing map and ranked tensor type, this helper
    /// determines which tensor dimensions are sharded versus replicated.
    ///
    /// For each tensor dimension `d`, the dimension is marked as sharded if:
    ///   - The indexing map result for `d` is an AffineDimExpr, and
    ///   - That dimension expression refers exactly to `plan.shardIter`.
    ///
    /// In other words, a tensor dimension is sharded iff it is indexed
    /// directly by the loop iterator selected for sharding.
    ///
    /// The result is an ArrayAttr of length equal to the tensor rank, where:
    ///   - An empty GridAxesAttr (`[]`) denotes replication.
    ///   - A GridAxesAttr containing `[0]` denotes sharding along grid axis 0.
    ///
    /// Even if all dimensions are replicated, an explicit rank-sized array
    /// is returned.
    auto buildSplitAxesForOperand = [&](AffineMap map,
                                        RankedTensorType rtt) -> ArrayAttr {
      SmallVector<Attribute> perDimAxes;
      perDimAxes.reserve(rtt.getRank());

      // map: (loops...) -> (tensor_dims...)
      for (int64_t d = 0; d < rtt.getRank(); ++d) {
        bool splitThisDim = false;

        // Only consider direct dim expressions for now.
        if (d < (int64_t)map.getNumResults()) {
          if (auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(d))) {
            if ((int64_t)dimExpr.getPosition() == plan.shardIter)
              splitThisDim = true;
          }
        }

        if (splitThisDim) {
          // Shard this tensor dimension on grid axis 0.
          perDimAxes.push_back(
              shard::GridAxesAttr::get(ctx, ArrayRef<int64_t>{0}));
        } else {
          // Replicated tensor dimension.
          perDimAxes.push_back(
              shard::GridAxesAttr::get(ctx, ArrayRef<int64_t>{}));
        }
      }

      // Note: Even if all entries are empty, we still return an array of size
      // 'rank' (e.g. `[[], [], ...]`), which is a valid "replicated" encoding.
      return ArrayAttr::get(ctx, perDimAxes);
    };

    /// Materialize a `!shard.sharding` SSA value for a tensor operand.
    ///
    /// This helper constructs an explicit sharding descriptor as a value,
    /// which can later be consumed by `shard.shard` ops and propagated through
    /// use-def chains.
    ///
    /// Behavior:
    ///  - If the operand is not a RankedTensorType, no sharding is created and
    ///    an empty Value is returned.
    ///  - Otherwise, the operand's indexing map is analyzed to compute
    ///    `split_axes` via `buildSplitAxesForOperand`.
    ///  - A `shard::ShardingOp` is created referencing the provided grid symbol.
    ///
    /// Halo sizes and per-dimension offsets are intentionally left empty;
    /// they are not inferred at this stage of planning.
    ///
    /// \returns A Value of type `!shard.sharding`, or an empty Value if the
    ///          operand cannot be sharded.
    auto buildShardingValueFor = [&](AffineMap map, Value tensorVal) -> Value {
      auto rtt = dyn_cast<RankedTensorType>(tensorVal.getType());
      if (!rtt)
        return Value();

      // Build split_axes = [ GridAxesAttr, GridAxesAttr, ... ] (rank entries).
      ArrayAttr splitAxesAttr = buildSplitAxesForOperand(map, rtt);

      // shard.sharding takes a symbol ref to the grid (e.g. @nsp).
      auto gridRef = FlatSymbolRefAttr::get(grid.getSymNameAttr());

      // Create the sharding descriptor value.
      // NOTE: halo sizes / offsets are kept empty for now.
      auto shardingOp = b.create<shard::ShardingOp>(
          loc,
          /*grid=*/gridRef,
          /*split_axes=*/splitAxesAttr,
          /*static_halo_sizes=*/ArrayRef<int64_t>{},
          /*static_sharded_dims_offsets=*/ArrayRef<int64_t>{});

      return shardingOp.getResult(); // !shard.sharding
    };

    // -------------------------------------------------------------------
    // Main body
    // -------------------------------------------------------------------

    // Indexing maps are ordered as: inputs..., outputs...
    auto maps = op.getIndexingMapsArray();

    // Wrap each tensor input with shard.shard.
    SmallVector<Value> newInputs;
    newInputs.reserve(op.getNumInputs());
    for (unsigned i = 0; i < op.getNumInputs(); ++i) {
      Value in = op.getInputs()[i];
      AffineMap map = maps[i];

      Value shardingVal = buildShardingValueFor(map, in);
      if (!shardingVal) {
        newInputs.push_back(in);
        continue;
      }

      auto shardOp = b.create<shard::ShardOp>(
          loc,
          in.getType(),
          /*src=*/in,
          /*sharding=*/shardingVal,
          /*annotate_for_users=*/UnitAttr::get(ctx));
      newInputs.push_back(shardOp.getResult());
    }

    // Wrap each tensor output (init tensor) with shard.shard.
    SmallVector<Value> newOutputs;
    newOutputs.reserve(op.getNumOutputs());
    for (unsigned oi = 0; oi < op.getNumOutputs(); ++oi) {
      Value out = op.getOutputs()[oi];
      AffineMap map = maps[op.getNumInputs() + oi];

      Value shardingVal = buildShardingValueFor(map, out);
      if (!shardingVal) {
        newOutputs.push_back(out);
        continue;
      }

      auto shardOp = b.create<shard::ShardOp>(
          loc,
          out.getType(),
          /*src=*/out,
          /*sharding=*/shardingVal,
          /*annotate_for_users=*/UnitAttr::get(ctx));
      newOutputs.push_back(shardOp.getResult());
    }

    // Rebuild the linalg::GenericOp so that it consumes the sharded SSA values.
    //
    // Since sharding is represented as SSA data, simply inserting shard.shard
    // ops is insufficient. To ensure the computation consumes them, we must
    // create a new linalg.generic with (newInputs/newOutputs), then move the
    // region and replace the old op.
    b.setInsertionPoint(op);

    SmallVector<AffineMap> indexingMaps(op.getIndexingMapsArray());
    SmallVector<utils::IteratorType> iteratorTypes(op.getIteratorTypesArray());

    auto newOp = b.create<linalg::GenericOp>(
        loc,
        op->getResultTypes(),
        /*inputs=*/newInputs,
        /*outputs=*/newOutputs,
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes);

    // Preserve any extra attributes on the original generic op.
    // Avoid overwriting the structural attributes that define the generic itself.
    for (auto na : op->getAttrs()) {
      auto name = na.getName();
      if (name == linalg::GenericOp::getIndexingMapsAttrName() ||
          name == linalg::GenericOp::getIteratorTypesAttrName())
        continue;
      newOp->setAttr(name, na.getValue());
    }

    // Move the region body (no cloning) into the new generic.
    newOp.getRegion().takeBody(op.getRegion());

    // Replace all uses and erase the old op.
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();

    return newOp;
  }

  /// Insert an all-reduce if the sharding plan requires it.
  ///
  /// This is relevant for global reductions (e.g. sum-reduction) where each
  /// NSP produces a partial result and the program expects a replicated final value.
  static void materializeAllReduceIfNeeded(linalg::GenericOp op,
                                          const ShardPlan &plan) {
    // Skeleton:
    //  * Identify the value that represents the partial reduction.
    //  * Insert shard.all_reduce with op="add" (or other reduction kind).
    //
    // In many systems, this is done by a separate "collective materialization"
    // pass; keeping it explicit here helps debugging and documentation.
    (void)op;
    (void)plan;
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass registration.
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNSPShardPlannerPass() {
  return std::make_unique<NSPShardPlannerPass>();
}
