include_guard(GLOBAL)

set(targetName gameGui)

find_package(Qt6 REQUIRED COMPONENTS Widgets Multimedia)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/resources.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/boardWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/chatWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/gameWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/soundPlayer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardRenderer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/playerCard.hpp"
)

set(sources
	"${CMAKE_CURRENT_LIST_DIR}/resources.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/chatWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/gameWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/soundPlayer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardRenderer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/playerCard.cpp"
)

add_library(${targetName} STATIC ${headers} ${sources})
add_library(tengen::game::gui ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
)

target_link_libraries(${targetName}
	PUBLIC
		tengen::game::model
		Qt6::Widgets
		Qt6::Multimedia
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER  "${IDE_FOLDER_SOURCE}/game"
		AUTOMOC ON # Auto handling of QT Meta-Object Compiler
)

set(resourcesDir "${CMAKE_CURRENT_LIST_DIR}/resources")
target_compile_definitions(${targetName}
	PRIVATE
		GUI_RESOURCES_DIR="${resourcesDir}"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
