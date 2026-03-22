# Setup project settings
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Add libraries
message(STATUS "Setting up remote: gtest")
include("${CMAKE_CURRENT_SOURCE_DIR}/gtest.cmake")

message(STATUS "Setting up remote: Logger")
include("${CMAKE_CURRENT_SOURCE_DIR}/logger.cmake")

message(STATUS "Setting up remote: asio")
include("${CMAKE_CURRENT_SOURCE_DIR}/asio.cmake")

message(STATUS "Setting up remote: nlohmann_json")
include("${CMAKE_CURRENT_SOURCE_DIR}/nlohmann_json.cmake")
