cmake_minimum_required(VERSION 3.14)

project(Tengen
	VERSION 0.5.0
	DESCRIPTION "A modular Go platform bridging physical and digital games."
	LANGUAGES CXX
)


# ----- Settings -----

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(TENGEN_BUILD_TESTS   "Create the unit tests for the project." ON)
option(TENGEN_BUILD_TOOLS   "Create the tools for the project." ON)
option(TENGEN_BUILD_SANDBOX "Create the tools for the project." ON)


# ----- Environment -----

set(TENGEN_QT_MIN_VERSION 6.8)     # Minimum Qt version.
set(TENGEN_OPENCV_MIN_VERSION 5.0) # Minimum OpenCV version, any 5.x.

# Install layout: Windows keeps everything next to tengen.exe, Linux uses the GNUInstallDirs folders (bin/, ...).
if(WIN32)
	set(CMAKE_INSTALL_BINDIR ".")
endif()
include(GNUInstallDirs)

# License texts: next to tengen.exe on Windows, share/licenses/tengen on Linux (Arch and Fedora convention).
if(WIN32)
	set(TENGEN_INSTALL_LICENSEDIR "licenses")
else()
	set(TENGEN_INSTALL_LICENSEDIR "${CMAKE_INSTALL_DATAROOTDIR}/licenses/tengen")
endif()

# Setup compile_commands.json for clangd support (only Ninja and Makefile generators - no VS). use scripts/win/2_configure-clangd.ps1 on windows.
if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
	file(CREATE_LINK
		"${CMAKE_BINARY_DIR}/compile_commands.json"
		"${CMAKE_SOURCE_DIR}/compile_commands.json"
		RESULT TENGEN_COMPILE_COMMANDS_LINK_RESULT # Store result in var so cmake does not exit with failure
		SYMBOLIC
	)
endif()


# ----- Add Projects -----

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/lib")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/src")

if(TENGEN_BUILD_TOOLS)
	add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tools")
endif()
if(TENGEN_BUILD_TESTS)
	include(CTest)
	enable_testing()
	add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
endif()


# ----- Installation -----

# Licenses, laid out as NOTICES.md describes
# The resources folders bring the licenses of their third-party files along.
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/LICENSES/" DESTINATION "${TENGEN_INSTALL_LICENSEDIR}")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/NOTICES.md" DESTINATION "${TENGEN_INSTALL_LICENSEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/3rdparty/licenses/" DESTINATION "${TENGEN_INSTALL_LICENSEDIR}/3rdparty")

# The official OpenCV package for Windows brings the licenses of the libraries it contains.
if(WIN32 AND EXISTS "${OpenCV_DIR}/etc/licenses")
	install(DIRECTORY "${OpenCV_DIR}/etc/licenses/" DESTINATION "${TENGEN_INSTALL_LICENSEDIR}/3rdparty/opencv")
endif()
