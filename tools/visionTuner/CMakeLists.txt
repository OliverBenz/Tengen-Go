set(targetName visionTuner)

find_package(Qt6 REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/pipelineStep.hpp" 
	"${CMAKE_CURRENT_LIST_DIR}/analyser.hpp" 
	"${CMAKE_CURRENT_LIST_DIR}/mainWindow.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/main.cpp" 
	"${CMAKE_CURRENT_LIST_DIR}/analyser.cpp" 
	"${CMAKE_CURRENT_LIST_DIR}/mainWindow.cpp" 
)

add_executable(${targetName} ${headers} ${sources})

target_link_libraries(${targetName}
	PRIVATE
		tengen::vision::core
		Qt6::Widgets
)
target_compile_definitions(${targetName}
	PRIVATE
		PATH_TEST_IMG="${CMAKE_SOURCE_DIR}/tests/vision/resources/"
)

# Setup project settings
set_project_warnings(${targetName})  # Which warnings to enable
set_compile_options(${targetName})   # Which extra compiler flags to enable
set_output_directory(${targetName})  # Set the output directory of the library
