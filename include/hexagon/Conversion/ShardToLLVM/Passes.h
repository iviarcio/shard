//===- Passes.h - Convert Shard to LLVM ops ---------------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#ifndef HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H
#define HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H

#include "hexagon/Conversion/ShardToLLVM/ShardToLLVM.h"

namespace mlir {
namespace hexagon {

#define GEN_PASS_REGISTRATION
#include "hexagon/Conversion/ShardToLLVM/Passes.h.inc"

} // namespace hexagon
} // namespace mlir

#endif // HEXAGON_CONVERSION_SHARDTOLLVM_PASSES_H
