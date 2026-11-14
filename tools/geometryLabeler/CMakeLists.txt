include_guard(GLOBAL)

set(targetName geometryLabeler)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets)

set(headers
	"${CMAKE_CURRENT_LIST_DIR}/boardCanvas.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardFile.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/geometryFile.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/grid.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/labelPresenter.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/labelView.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/layoutPresenter.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/layoutView.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/photoCanvas.hpp"
	"${CMAKE_CURRENT_LIST_DIR}/rectify.hpp"
)
set(sources
	"${CMAKE_CURRENT_LIST_DIR}/main.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardCanvas.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/boardFile.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/geometryFile.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/grid.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/labelPresenter.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/labelView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/layoutPresenter.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/layoutView.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/photoCanvas.cpp"
	"${CMAKE_CURRENT_LIST_DIR}/rectify.cpp"
)

add_executable(${targetName} ${headers} ${sources})

target_link_libraries(${targetName}
	PRIVATE
		nlohmann_json::nlohmann_json
		Qt6::Widgets
)

set_target_properties(${targetName}
	PROPERTIES
		FOLDER  "${IDE_FOLDER_TOOLS}"
		AUTOMOC ON
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
tengen_deploy_qt(${targetName})
tengen_copy_runtime_dlls(${targetName})
