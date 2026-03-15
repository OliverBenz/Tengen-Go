set(targetName gameGui)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

# Get files to build
set(headers
    "${CMAKE_CURRENT_LIST_DIR}/include/gui/boardWidget.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/include/gui/chatWidget.hpp"
    "${CMAKE_CURRENT_LIST_DIR}/boardRenderer.hpp"
)

set(sources
    "${CMAKE_CURRENT_LIST_DIR}/boardWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/chatWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/boardRenderer.cpp"
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
)

set_target_properties(${targetName} PROPERTIES AUTOMOC ON) # Auto handling of QT Meta-Object Compiler
target_compile_definitions(${targetName} PRIVATE TEXTURE_BLACK="${CMAKE_SOURCE_DIR}/assets/anime_black.png")
target_compile_definitions(${targetName} PRIVATE TEXTURE_WHITE="${CMAKE_SOURCE_DIR}/assets/anime_white.png")


# Setup project settings
set_project_warnings(${targetName})
set_compile_options(${targetName})
set_output_directory(${targetName})
