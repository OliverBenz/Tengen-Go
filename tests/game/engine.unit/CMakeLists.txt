include_guard(GLOBAL)

set(targetName gameEngine.unit)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/gnuGo.gtest.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/kataGo.gtest.cpp"
)

# Link to required libraries
target_link_libraries(${targetName}
    PRIVATE
        tengen::game::engine
        GTest::gtest_main
)

set_target_properties(${targetName}
     PROPERTIES
         FOLDER "${IDE_FOLDER_TESTS}/game"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})

# Add tests
include(GoogleTest)
gtest_discover_tests(${targetName})
