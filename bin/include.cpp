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

