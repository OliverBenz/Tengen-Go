include_guard(GLOBAL)

set(targetName visionGui)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/manualSetupView.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/vision/gui/manualSetupPresenter.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/imageView.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/manualSetupView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/manualSetupPresenter.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/imageView.cpp"
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
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER  "${IDE_FOLDER_SOURCE}/vision"
		AUTOMOC ON # Auto handling of QT Meta-Object Compiler
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
