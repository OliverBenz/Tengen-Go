include_guard(GLOBAL)

set(targetName gameEngine)

# Create target
add_library(${targetName} STATIC
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/botMove.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/kataGo.hpp"

	"${CMAKE_CURRENT_LIST_DIR}/kataGo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/gtp.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/gtp.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/subProcess.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/subProcess.cpp"
)
add_library(tengen::game::engine ALIAS ${targetName})

target_include_directories(${targetName}
	PUBLIC
		"${CMAKE_CURRENT_LIST_DIR}/include"
)
target_link_libraries(${targetName}
	PUBLIC
		tengen::game::model
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/game"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})  # Set the output directory of the library
