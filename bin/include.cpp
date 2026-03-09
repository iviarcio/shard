/* In linalgToLLVMPass.cpp do: */

#include "hexagon/NSP/NSPPasses.h" 
#include "hexagon/Conversion/ShardToLLVM/Passes.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"

/* Include in getDependentDialects(...) */

registry.insert<mlir::shard::ShardDialect>();

/* include before if (enableBufferization) */

// ---- NSP sharding + SPMDization ----
// Run after most high-level linalg transforms, but before any bufferization
// to keep sharding decisions consistent with the final tensor schedule.
if (enableNSPSharding) {
  pm.addNestedPass<func::FuncOp>(
      mlir::hexagon::createNSPShardPlannerPass(/*nspCount=*/nspCount,
                                              /*allowCollectives=*/allowCollectives));

  pm.addNestedPass<func::FuncOp>(mlir::shard::createShardingPropagation());

  pm.addPass(mlir::hexagon::createNSPSpmdizePass(/*allowCollectives=*/allowCollectives));

  // Eliminate shard dialect ops before bufferization/lowering.
  pm.addPass(mlir::hexagon::createShardToLLVMPass());

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

/* in Passes.td from LinaglToLLVM do: */

Option<"enableNSPSharding", "enable-nsp-sharding", "bool",
       /*default=*/"false",
       "Enable NSP sharding (planner + propagation + spmdize).">,

Option<"nspCount", "nsp-count", "int",
       /*default=*/"16",
       "Number of NSP devices to shard across.">,

Option<"allowCollectives", "allow-collectives", "bool",
       /*default=*/"false",
       "Allow collectives during NSP sharding/spmdize (otherwise restrict to distributed loops only).">,


// /* in hexagon_options.py (/bin/python):
# NSP sharding options
enableNSPSharding: bool = False
nspCount: int = 16
allowCollectives: bool = False

Add NSPSharding infrastructure and initial integration with Hexagon flow

This commit introduces the NSPSharding infrastructure and its initial
integration into the Hexagon MLIR compilation flow.

NSPSharding provides a framework to annotate and propagate tensor
partitioning across operations targeting the NSP architecture. The
current implementation focuses on pre-tiling (tensor sharding) to
enable SPMD-style execution across multiple NPUs.

The infrastructure includes:
- NSPShardPlanner: analysis and insertion of shard annotations.
- Sharding interface models for operations not defined in the shard dialect.
- Passes for validation and propagation of sharding metadata.
- Initial lowering support to remove shard annotations before LLVM lowering.

The design aims to be general and extensible so that sharding can be
incrementally supported across a wider set of workloads and operators.

Current limitations:
- Communication collectives (e.g., all_reduce) are not yet lowered.
- Reductions across partitioned dimensions require future collective support.

This change establishes the foundation for distributed tensor execution
on NSP devices and will be extended as additional workloads are enabled.


======= Review:

**1. Validation in `getLocalType` (NSPSPmdize.cpp)**
Partially applicable. `getLocalType` currently checks that `dim0 % numShards == 0`, which is sufficient for this stage of the pipeline. The NSP sharding configuration (grid shape and split axes) is determined earlier by `NSPShardPlanner`, so `NSPSpmdize` assumes a valid sharding plan. Additional global validation of grid compatibility is therefore outside the responsibility of this pass.
That said, improving diagnostics (e.g., clearer error/warning messages for dynamic or non-divisible dimensions) would be reasonable.

if (ShapedType::isDynamic(dim0)) {
  module.emitWarning()
      << "NSPSpmdize: dynamic dimension cannot be statically sharded";
  return RankedTensorType();
}

**2. Error handling in `distributeScfForCyclic`**
Most of the suggested checks are not applicable in the MLIR pass model. Builders such as `create<scf::ForOp>` and `clone()` do not return failure values; invalid IR is caught later by verifiers. Therefore checks such as validating the result of `clone()` or testing for null operations are unnecessary.
A small improvement that may be considered is validating that the loop step is positive when statically known.

if (auto c = step.getDefiningOp<arith::ConstantIndexOp>())
  if (c.value() <= 0)
     return;

**3. “Race condition” in the NSP sharding pipeline**
This concern does not apply. MLIR pass pipelines execute sequentially and each pass operates on a stable IR state. There is no concurrent mutation of the IR between passes.
Running canonicalization or CSE between passes could be useful for IR cleanup, but it is not required for correctness.

**4. Redundant IR walks in `NSPShardPlanner`**
This observation is reasonable but not critical. Collecting operations in a worklist before transforming them is a common MLIR pattern when transformations may modify the IR. While the walk could potentially be optimized, the current structure is intentional and safe.

**5. Hardcoded ABI assumptions in `ShardToLLVMPass`**
This is a valid concern. The current implementation relies on fixed argument positions for runtime values used to compute the linear process index. Adding explicit validation of the expected argument count and types would improve robustness if the ABI changes in the future.

In summary, the ABI validation point is the most relevant improvement. The other suggestions either fall outside the scope of the pass (planner-level validation) or reflect patterns that are typical in MLIR pass implementations.

---
