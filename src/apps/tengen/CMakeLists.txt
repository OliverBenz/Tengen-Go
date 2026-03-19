set(targetName tengen)

find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/MainWindow.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/MainWindowPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/GamePresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/BoardPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ChatPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ConnectDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/HostDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Logging.cpp"
)

target_link_libraries(${targetName} 
    PRIVATE
        Logger::Logger
        tengen::game::runtime
        tengen::game::gui
        Qt6::Widgets
)

set_target_properties(${targetName} PROPERTIES AUTOMOC ON) # Auto handling of QT Meta-Object Compiler
target_compile_definitions(${targetName} PRIVATE TEXTURE_BLACK="${CMAKE_SOURCE_DIR}/assets/anime_black.png")
target_compile_definitions(${targetName} PRIVATE TEXTURE_WHITE="${CMAKE_SOURCE_DIR}/assets/anime_white.png")

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
