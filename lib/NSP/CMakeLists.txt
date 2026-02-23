add_mlir_library(MLIRNSP
  NSPShardInterfaceModels.cpp
  NSPShardPlanner.cpp
  NSPSpmdize.cpp
  NSPPasses.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRShardDialect
    MLIRSupport
)
