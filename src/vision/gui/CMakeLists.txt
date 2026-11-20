include_guard(GLOBAL)

set(targetName visionGui)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets Multimedia)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/manualSetupView.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/manualSetupPresenter.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/cameraSource.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/cameraPreviewView.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/cameraPreviewPresenter.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/imageView.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/src/manualSetupView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/manualSetupPresenter.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/imageView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/cameraSource.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/cameraPreviewView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/cameraPreviewPresenter.cpp"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::vision::gui ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
)
target_link_libraries(${targetName}
	PUBLIC
		tengen::vision::perception
		Qt6::Widgets
		Qt6::Multimedia # cameraSource.hpp holds the camera.
)

# Debug builds offer this photo as a camera in the preview, to play without a webcam.
target_compile_definitions(${targetName}
	PRIVATE
		TENGEN_TEST_IMAGE="${CMAKE_SOURCE_DIR}/tests/vision/resource/angled_easy/angle_1.jpeg"
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER  "${IDE_FOLDER_SOURCE}/vision"
		AUTOMOC ON # Auto handling of QT Meta-Object Compiler
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
