include_guard(GLOBAL)

set(targetName gameGui)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets Multimedia)

# Get files to build
set(headers
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/resources.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/boardWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/chatWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/gameWidget.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/gui/soundPlayer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardRenderer.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/playerCard.hpp"
)

set(sources
	"${CMAKE_CURRENT_LIST_DIR}/src/resources.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/chatWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/gameWidget.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/soundPlayer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/boardRenderer.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/src/playerCard.cpp"
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

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})

# Copy resource directory to the output
add_custom_target(${targetName}_resource ALL
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
	        "${CMAKE_CURRENT_LIST_DIR}/resource"
	        "$<GENEX_EVAL:$<TARGET_PROPERTY:${targetName},RUNTIME_OUTPUT_DIRECTORY>>/resource"
)
add_dependencies(${targetName} ${targetName}_resource)
set_target_properties(${targetName}_resource
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/game"
)

# ----- Installation -----
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/resource/" DESTINATION "${CMAKE_INSTALL_BINDIR}/resource")
