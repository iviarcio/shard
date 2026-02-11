//===- NSPPasses.cpp - Sharding and SPMDization Passes in MLIR ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Registration glue for the NSP "shard -> SPMD" pipeline.
//
// This module sits in lib/NSP/ and typically owns:
//   1) Registering NSP passes (planner + spmdization/materialization).
//   2) Registering other pass pipelines (for mlir-opt).
//   3) Registering dialect extensions to attach ShardingInterface models
//      (so sharding-propagation can traverse "boundary/view/sink" ops).
//
//   - NSPShardPlanner.cpp
//   - NSPShardInterfaceModels.cpp
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

// Useful common transforms.
#include "mlir/Transforms/Passes.h"        // canonicalizer, cse

// shard dialect IR (GridOp, ShardOp, collectives, attrs).
#include "mlir/Dialect/Shard/Transforms/Passes.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/Dialect/Shard/IR/ShardDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "hexagon/NSP/NSPPasses.h"

// NSP pieces.
namespace mlir {
namespace hexagon {

std::unique_ptr<Pass> createNSPShardPlannerPass();
std::unique_ptr<Pass> createNSPSpmdizePass();

// From NSPShardInterfaceModels.cpp (attach external models).
void registerNSPShardInterfaceModels(DialectRegistry &registry);

} // namespace hexagon
} // namespace mlir

using namespace mlir;
using namespace hexagon;

namespace {

// Pipeline options:
//   -nsp-shard="nsp-count=16 allow-collectives"
struct NSPShardPipelineOptions
    : public PassPipelineOptions<NSPShardPipelineOptions> {
  Option<int64_t> nspCount{*this, "nsp-count",
                           llvm::cl::desc("Number of NSP devices in the mesh"),
                           llvm::cl::init(16)};

  Option<bool> runPropagation{
      *this, "propagate",
      llvm::cl::desc("Run sharding propagation after planning"),
      llvm::cl::init(true)};

  Option<bool> allowCollectives{
      *this, "allow-collectives",
      llvm::cl::desc(
          "Allow planner/SPMDization to insert collectives (e.g. all-reduce)"),
      llvm::cl::init(false)};

  Option<bool> canonicalize{
      *this, "canonicalize",
      llvm::cl::desc("Run canonicalize + CSE at key boundaries"),
      llvm::cl::init(true)};
};

/// Build the canonical NSP shard pipeline. At high-level:
///  1. Planner: attach shard annotations (device mesh + tensor layout).
///  2. Propagation: infer missing shardings using ShardingInterface.
///  3. SPMDization: lower shard IR to explicit slices + collectives.
///  4. Cleanup: canonicalize/CSE.
static void buildNSPShardPipeline(OpPassManager &pm,
                                  const NSPShardPipelineOptions &opts) {

  // 1. Planner (NSPShardPlannerPass in NSPShardPlanner.cpp)
  // To Planner options (nsp-count / allow-collectives) reach the
  // pass, it exposes as pass options and plumb here.
  pm.addPass(createNSPShardPlannerPass());
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // 2. Sharding propagation (Shard dialect pass)
  // This pass relies on shard::ShardingInterface models for ops in the graph.
  // Our NSPShardInterfaceModels.cpp attaches missing models.
  if (opts.runPropagation) {
    pm.addPass(shard::createShardingPropagation());
  }
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }

  // 3. SPMDization / materialization
  // This pass is expected to:
  //   - compute per-NSP tensor slices (extract_slice/subview),
  //   - materialize collectives (all-reduce/all-gather/all-to-all),
  //   - rewrite compute ops to operate on local tiles.

  // pm.addPass(createNSPSpmdizePass());

  // 4. Cleanup
  if (opts.canonicalize) {
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// Public registration entrypoints.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hexagon {

/// Register NSP passes (so tools can do: -nsp-shard-planner -nsp-spmdize ...).
///
/// In the future we'll using TableGen, i.e., auto-generate these via
/// Passes.td and call registerNSPPasses() from that generated file.
/// For now, we use an explicit style that works well for tests.
void registerNSPPasses() {
  // The lambda form avoids needing the pass type visible here.
  registerPass([]() -> std::unique_ptr<Pass> { return createNSPShardPlannerPass(); });
  registerPass([]() -> std::unique_ptr<Pass> { return createNSPSpmdizePass(); });
}

/// Register NSP pipelines.
void registerNSPPipelines() {
  PassPipelineRegistration<NSPShardPipelineOptions>(
      "nsp-shard",
      "NSP pipeline: shard planning -> sharding propagation -> SPMDization",
      buildNSPShardPipeline);
}

/// Register dialect extensions required by NSP sharding propagation.
void registerNSPDialectExtensions(DialectRegistry &registry) {
  registry.insert<mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::shard::ShardDialect>();

  registerNSPShardInterfaceModels(registry);
}

} // namespace hexagon
} // namespace mlir
