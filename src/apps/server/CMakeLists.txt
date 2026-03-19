set(targetName server)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
)

target_link_libraries(${targetName}
    PRIVATE
        tengen::game::runtime
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
