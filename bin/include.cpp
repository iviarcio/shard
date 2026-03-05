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
