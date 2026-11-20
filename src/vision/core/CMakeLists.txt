include_guard(GLOBAL)

set(targetName visionCore)

find_package(OpenCV ${TENGEN_OPENCV_MIN_VERSION} REQUIRED COMPONENTS
	core
	imgproc
	imgcodecs
	videoio
	highgui
	geometry
	dnn
)

# Compile the stone model in.
set(stoneNetModel "${CMAKE_CURRENT_LIST_DIR}/resource/models/stoneNet.onnx")
set(stoneNetSource "${CMAKE_CURRENT_BINARY_DIR}/generated/stoneNetModel.cpp")
add_custom_command(
	OUTPUT "${stoneNetSource}"
	COMMAND ${CMAKE_COMMAND}
	        -DINPUT=${stoneNetModel}
	        -DOUTPUT=${stoneNetSource}
	        -DHEADER=stoneNetModel.hpp
	        -DNAMESPACE=tengen::vision::core
	        -DFUNCTION=stoneNetModel
	        -P "${CMAKE_CURRENT_LIST_DIR}/cmake/embedFile.cmake"
	DEPENDS "${stoneNetModel}" "${CMAKE_CURRENT_LIST_DIR}/cmake/embedFile.cmake"
	COMMENT "Compiling in stoneNet.onnx"
	VERBATIM
)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/src/statistics.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridLatticeFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinderInternal.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/debugVisualizer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/boardFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/gridFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/stoneClassifier.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/stoneFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneNetModel.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/src/statistics.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/debugVisualizer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinderInternal.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridLatticeFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneClassifier.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder.cpp"
	"${stoneNetSource}"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::vision::core ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
		${OpenCV_INCLUDE_DIRS}
	PRIVATE
		"${CMAKE_CURRENT_LIST_DIR}/src" # For the generated model source.
)
target_link_libraries(${targetName}
	PUBLIC
		${OpenCV_LIBS}
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/vision"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})

target_compile_definitions(${targetName}
	PRIVATE
		$<$<CONFIG:Debug>:VISION_DEBUG_LOGGING>
		# VISION_LOG_BOARDFINDER
		# VISION_LOG_GRIDFINDER
)
