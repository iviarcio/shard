//===- NSPDialect.cpp -----------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Private NSP dialect used by the Hexagon NSP pipeline.
//
//===----------------------------------------------------------------------===//

#include "NSP/IR/NSPDialect.h"
#include "NSP/IR/NSPOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shard/IR/ShardOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::hexagon::nsp;

#include "NSP/IR/NSPDialect.cpp.inc"

void NSPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NSP/IR/NSPOps.cpp.inc"
      >();
}
