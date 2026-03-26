add_mlir_library(MLIRNSP
  NSPShardInterfaceModels.cpp
  NSPShardPlanner.cpp
  NSPLocalizePass.cpp
  NSPMaterializePass.cpp
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
