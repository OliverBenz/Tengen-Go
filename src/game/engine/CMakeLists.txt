include_guard(GLOBAL)

set(targetName gameEngine)

# The subprocess transport is the only platform specific part of the engine.
if(WIN32)
	set(subProcessSource "${CMAKE_CURRENT_LIST_DIR}/subProcess_windows.cpp")
else()
	set(subProcessSource "${CMAKE_CURRENT_LIST_DIR}/subProcess_linux.cpp")
endif()

# Create target
add_library(${targetName} STATIC
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/botMove.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/IEngineListener.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/engineCatalog.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/gtpEngine.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/kataGo.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/kataGoConfig.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/gnuGo.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/include/engine/gnuGoConfig.hpp"

	"${CMAKE_CURRENT_LIST_DIR}/engineCatalog.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/gtpEngine.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/kataGo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/gnuGo.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/gtp.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/gtp.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/subProcess.hpp"
	"${subProcessSource}"
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
if(WIN32)
	set_target_configuration(${targetName} Config::DefaultWin)
else()
	set_target_configuration(${targetName} Config::Default)
endif()

set_output_directory(${targetName})  # Set the output directory of the library
