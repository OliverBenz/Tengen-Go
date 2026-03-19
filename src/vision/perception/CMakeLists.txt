set(targetName visionPerception)

# TODO: We still need this if tengen::vision::core links OpenCV privately?
find_package(OpenCV REQUIRED)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/source.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/vision.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/vision.cpp"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::vision::perception ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
        "${CMAKE_CURRENT_LIST_DIR}/include"
)
target_link_libraries(${targetName}
	PRIVATE
		tengen::vision::core
	PUBLIC
		tengen::game::model
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
