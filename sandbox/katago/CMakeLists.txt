include_guard(GLOBAL)

set(targetName sbKatago)
add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/subProcess.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/kataGo.cpp"
)

target_compile_definitions(${targetName}
	PRIVATE
		KATAGO_PATH="${CMAKE_CURRENT_LIST_DIR}/katago_final"
)

target_link_libraries(${targetName}
    PRIVATE
        tengen::game::model
)

# Setup project settings
# set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
