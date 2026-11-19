include_guard(GLOBAL)

set(targetName visionCore)

find_package(OpenCV ${TENGEN_OPENCV_MIN_VERSION} REQUIRED COMPONENTS
	core
	imgproc
	imgcodecs
	videoio
	highgui
	geometry
)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/src/statistics.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridLatticeFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinderInternal.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/debugVisualizer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/boardFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/gridFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/stoneFinder.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/core/stoneFinderConfig.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/decisionPolicy.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/debugging.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/featureExtraction.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/features.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/geometrySampling.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/modelCalibration.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/refinementEngine.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/scoring.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/stoneFinderInternal.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/stoneFinderTypes.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/src/statistics.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/debugVisualizer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinderInternal.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridLatticeFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gridFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/debugging.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/decisionPolicy.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/featureExtraction.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/geometrySampling.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/modelCalibration.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/refinementEngine.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/scoring.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/stoneFinder.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/stoneFinder/stoneFinderInternal.cpp"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::vision::core ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
		${OpenCV_INCLUDE_DIRS}
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
		# VISION_LOG_STONEFINDER
)
