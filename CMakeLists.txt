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
