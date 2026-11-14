add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/visionTuner/") # Application: Vision Paramter Tuner
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/boardViewer/") # Application: Render go board from different inputs
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/manualSetup/") # Application: Pick the board corners by hand, then run the vision pipeline
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/geometryLabeler/") # Application: Click the corners of photos to write their geometry .json
