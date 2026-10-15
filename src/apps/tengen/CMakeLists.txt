include_guard(GLOBAL)

set(targetName tengen)

find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/Logging.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/presenters/MainWindowPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/presenters/GamePresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/presenters/BoardPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/presenters/ChatPresenter.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/views/MainWindow.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/dialogs/LocalGameDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/BotDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/ConnectDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/HostDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/HelpDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/AboutDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dialogs/BoardStyleDialog.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/controls/BoardSizeWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controls/PlayerColourWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controls/RulesConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controls/GnuGoConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/controls/KataGoConfigWidget.cpp"
)

target_include_directories(${targetName}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/presenters"
        "${CMAKE_CURRENT_LIST_DIR}/views"
        "${CMAKE_CURRENT_LIST_DIR}/dialogs"
        "${CMAKE_CURRENT_LIST_DIR}/controls"
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
            "${CMAKE_CURRENT_LIST_DIR}/resources/help"
            "$<GENEX_EVAL:$<TARGET_PROPERTY:${targetName},RUNTIME_OUTPUT_DIRECTORY>>/help"
)
add_dependencies(${targetName} ${targetName}_help)
set_target_properties(${targetName}_help
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/apps"
)
