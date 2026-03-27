add_mlir_dialect_library(MLIRNSP
  NSPDialect.cpp
  NSPOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/hexagon/NSP/IR

  DEPENDS
    MLIRNSPDialectIncGen
    MLIRNSPOpsIncGen

  LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRBufferizationDialect
    MLIRIR
    MLIRMemRefDialect
    MLIRShardDialect
    MLIRSupport
)
