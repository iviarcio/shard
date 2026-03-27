//===- NSPOps.h -----------------------------------------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Private NSP operations used by the Hexagon NSP pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef QCOM_HEXAGON_BACKEND_NSP_IR_NSPOPS_H
#define QCOM_HEXAGON_BACKEND_NSP_IR_NSPOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/SymbolInterfaces.h"

#include "NSP/IR/NSPDialect.h"

#define GET_OP_CLASSES
#include "NSP/IR/NSPOps.h.inc"

#endif // QCOM_HEXAGON_BACKEND_NSP_IR_NSPOPS_H
