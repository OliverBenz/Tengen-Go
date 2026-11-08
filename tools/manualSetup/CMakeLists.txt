include_guard(GLOBAL)

set(targetName manualSetup)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets)

add_executable(${targetName}
	"${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries(${targetName}
	PRIVATE
		tengen::vision::gui
		Qt6::Widgets
)
set_target_properties(${targetName}
	PROPERTIES
		FOLDER "${IDE_FOLDER_TOOLS}"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
tengen_deploy_qt(${targetName})
tengen_copy_runtime_dlls(${targetName})
