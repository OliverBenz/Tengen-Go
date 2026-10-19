# Handling QT6 and OpenCV5 for Windows and Linux
# ----------------------------------------------
# Windows: Lets find_package find the Qt and OpenCV installs of scripts/win/1_setup_deps.py, so configuring needs no paths.
# Linux:   Qt and OpenCV come from the system packages, find_package finds them by itself.

# Copy the Qt DLLs and plugins next to a Qt executable.
function(tengen_deploy_qt targetName)
	if(WIN32)
		add_custom_command(TARGET ${targetName} POST_BUILD
			COMMAND Qt6::windeployqt --no-translations --no-compiler-runtime --verbose 0 "$<TARGET_FILE:${targetName}>"
		)
	endif()
endfunction()

# Copy the DLLs of all linked imported libraries  next to an executable.
function(tengen_copy_runtime_dlls targetName)
	if(WIN32)
		add_custom_command(TARGET ${targetName} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:${targetName}> "$<TARGET_FILE_DIR:${targetName}>"
			COMMAND_EXPAND_LISTS
		)
	endif()
endfunction()


# On Linux, we use system packages
if(NOT WIN32)
	return()
endif()


# ----- Settings -----

if(DEFINED ENV{TENGEN_DEPS_DIR})
	file(TO_CMAKE_PATH "$ENV{TENGEN_DEPS_DIR}" defaultDepsDir)
else()
	file(TO_CMAKE_PATH "$ENV{LOCALAPPDATA}/tengen/deps" defaultDepsDir)
endif()

set(TENGEN_DEPS_DIR "${defaultDepsDir}"  CACHE PATH   "Install folder of scripts/win/1_setup_deps.py.")
set(TENGEN_QT_ARCH  "win64_msvc2022_64" CACHE STRING "Qt architecture, see scripts/win/1_setup_deps.py --qt-arch.")


# ----- Search paths -----
# Layout: <deps>/qt/<arch>/<version>/<folder>/lib (aqt names the last folder) and <deps>/opencv-<version>/opencv/build.
# Newest version first, so find_package prefers it when the script installed several.

file(GLOB qtLibs "${TENGEN_DEPS_DIR}/qt/${TENGEN_QT_ARCH}/*/*/lib")
list(TRANSFORM qtLibs REPLACE "/lib$" "")
file(GLOB opencvPrefixes "${TENGEN_DEPS_DIR}/opencv-*/opencv/build")

set(prefixes ${qtLibs} ${opencvPrefixes})
list(SORT prefixes COMPARE NATURAL ORDER DESCENDING)

# Included from lib/CMakeLists.txt: PARENT_SCOPE makes the paths visible to the whole project, not only lib/.
list(APPEND CMAKE_PREFIX_PATH ${prefixes})
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
