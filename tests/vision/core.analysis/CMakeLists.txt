include_guard(GLOBAL)

set(targetName visionCore.analysis)

set(headers
    "${CMAKE_CURRENT_LIST_DIR}/corpus.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/measure.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/report.hpp"
)
set(sources
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/corpus.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/measure.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/report.cpp"
)

add_executable(${targetName} ${headers} ${sources})

target_link_libraries(${targetName}
    PRIVATE
        tengen::vision::core
        nlohmann_json::nlohmann_json
)

set_target_properties(${targetName}
    PROPERTIES
        FOLDER "${IDE_FOLDER_TESTS}/vision"
)

target_compile_definitions(${targetName}
    PRIVATE
        PATH_TEST_IMG="${CMAKE_CURRENT_LIST_DIR}/../resources/"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})

# Deliberately not registered with gtest_discover_tests(): this reports measurements, it does not assert.
