# Setup project settings
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add libraries
include("${CMAKE_CURRENT_SOURCE_DIR}/gtest.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/logger.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/asio.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json.cmake")
