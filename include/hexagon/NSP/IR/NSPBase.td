//===- NSPBase.td - NSP Dialect ----------------------------*- tablegen -*-===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// Private NSP dialect used as an IR contract between NSPLocalizePass and
// NSPMaterializePass. This dialect is intentionally local to hexagon-mlir
//
//===----------------------------------------------------------------------===//

#ifndef QCOM_HEXAGON_BACKEND_NSP_IR_NSPBASE_TD
#define QCOM_HEXAGON_BACKEND_NSP_IR_NSPBASE_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// NSP Dialect
//===----------------------------------------------------------------------===//

def NSP_Dialect : Dialect {
  let name = "nsp";
  let cppNamespace = "::mlir::hexagon::nsp";

  let description = [{
    Private dialect for the Hexagon NSP pipeline. It carries temporary IR
    contracts between NSP-specific passes, especially between localization and
    materialization, so that downstream tiling/vectorization can run without
    depending on placeholder global producers.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::shard::ShardDialect"
  ];
}

class NSP_Op<string mnemonic, list<Trait> traits = []> :
    Op<NSP_Dialect, mnemonic, traits> {
}

#endif // QCOM_HEXAGON_BACKEND_NSP_IR_NSPBASE_TD
