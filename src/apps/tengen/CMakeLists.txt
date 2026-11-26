include_guard(GLOBAL)

set(targetName tengen)

find_package(Qt6 ${TENGEN_QT_MIN_VERSION} REQUIRED COMPONENTS Widgets Concurrent)

add_executable(${targetName}
    "${CMAKE_CURRENT_LIST_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/Logging.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/src/presenters/MainWindowPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/presenters/GamePresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/presenters/BoardPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/presenters/ChatPresenter.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/presenters/NetworkPresenter.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/src/views/MainWindow.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/LocalGameDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/BotDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/ConnectDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/HostDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/HelpDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/AboutDialog.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/dialogs/StyleDialog.cpp"

    "${CMAKE_CURRENT_LIST_DIR}/src/controls/BoardSizeWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/controls/PlayerColourWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/controls/RulesConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/controls/GnuGoConfigWidget.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/controls/KataGoConfigWidget.cpp"
)

target_include_directories(${targetName}
    PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/src/presenters"
        "${CMAKE_CURRENT_LIST_DIR}/src/views"
        "${CMAKE_CURRENT_LIST_DIR}/src/dialogs"
        "${CMAKE_CURRENT_LIST_DIR}/src/controls"
)

target_link_libraries(${targetName}
    PRIVATE
        Logger::Logger
        tengen::game::runtime
        tengen::game::gui
        Qt6::Widgets
        Qt6::Concurrent
)

# The About dialog finds the installed license texts relative to the executable.
cmake_path(RELATIVE_PATH TENGEN_INSTALL_LICENSEDIR BASE_DIRECTORY "${CMAKE_INSTALL_BINDIR}" OUTPUT_VARIABLE licenseDir)
target_compile_definitions(${targetName}
    PRIVATE
        TENGEN_VERSION="${PROJECT_VERSION}"
        TENGEN_LICENSE_DIR="${licenseDir}"
)

set_target_properties(${targetName}
    PROPERTIES
      AUTOMOC          ON
      FOLDER           "${IDE_FOLDER_SOURCE}/apps"
      WIN32_EXECUTABLE ON
)

# Setup project settings
set_target_configuration(${targetName} Config::Default)
set_output_directory(${targetName})
tengen_deploy_qt(${targetName})


# Copy resource directory to the output
add_custom_target(${targetName}_resource ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
            "${CMAKE_CURRENT_LIST_DIR}/resource"
            "$<GENEX_EVAL:$<TARGET_PROPERTY:${targetName},RUNTIME_OUTPUT_DIRECTORY>>/resource"
)
add_dependencies(${targetName} ${targetName}_resource)
set_target_properties(${targetName}_resource
	PROPERTIES
		FOLDER "${IDE_FOLDER_SOURCE}/apps"
)


# ----- Installation -----
install(TARGETS ${targetName} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/resource/" DESTINATION "${CMAKE_INSTALL_BINDIR}/resource")

# Windows: ship the Qt DLLs and plugins, and the MSVC runtime, so it runs on machines without Qt or Visual Studio.
if(WIN32)
	# No FFmpeg:
	#   Qt plays our WAV sounds through its Windows backend, and FFmpeg would bring LGPL-2.1 source obligations.
	#   windeployqt has the options from Qt 6.12. Older ones deploy FFmpeg, which NOTICES.md doesn't cover.
	if(Qt6_VERSION VERSION_GREATER_EQUAL 6.12)
		set(deployOptions DEPLOY_TOOL_OPTIONS --no-ffmpeg --exclude-plugins ffmpegmediaplugin)
	else()
		install(CODE [[message(WARNING "This Qt deploys FFmpeg, which NOTICES.md doesn't cover. Package with Qt 6.12 or newer.")]])
	endif()

	qt_generate_deploy_app_script(TARGET ${targetName} OUTPUT_SCRIPT deployScript NO_TRANSLATIONS NO_COMPILER_RUNTIME ${deployOptions})
	install(SCRIPT "${deployScript}")

	set(CMAKE_INSTALL_SYSTEM_RUNTIME_DESTINATION "${CMAKE_INSTALL_BINDIR}")
	include(InstallRequiredSystemLibraries)
endif()
