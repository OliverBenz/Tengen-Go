include_guard(GLOBAL)

set(targetName tengen)

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/MainWindow.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/MainWindowPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/GamePresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/BoardPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/BoardSizeWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/LocalGameDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/RulesConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/BotDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/GnuGoConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/KataGoConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ChatPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/ConnectDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/HostDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/HelpDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/AboutDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/BoardStyleDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Logging.cpp"
)

target_link_libraries(${targetName}
    PRIVATE
        Logger::Logger
        tengen::game::runtime
        tengen::game::gui
        Qt6::Widgets
        Qt6::Concurrent
)

target_compile_definitions(${targetName}
    PRIVATE
        TENGEN_VERSION="${PROJECT_VERSION}"
)

set_target_properties(${targetName}
    PROPERTIES
      AUTOMOC ON # Auto handling of QT Meta-Object Compiler
      FOLDER "${IDE_FOLDER_SOURCE}/apps"
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})



add_custom_target(${targetName}_help ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${CMAKE_CURRENT_LIST_DIR}/help"
            "$<GENEX_EVAL:$<TARGET_PROPERTY:${targetName},RUNTIME_OUTPUT_DIRECTORY>>/help"
)
add_dependencies(${targetName} ${targetName}_help)
set_target_properties(${targetName}_help
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/apps"
)
