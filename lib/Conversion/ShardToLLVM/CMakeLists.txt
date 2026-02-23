add_mlir_conversion_library(ShardToLLVM
  ShardToLLVMPass.cpp

  DEPENDS
    ShardToLLVMConversionPassIncGen

  LINK_COMPONENTS
    Core

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRShardDialect
    MLIRTensorDialect
    MLIRTransforms
    MLIRSupport
    HexagonCommon
)
