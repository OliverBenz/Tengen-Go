# Lets find_package find the Qt and OpenCV installs of scripts/setup_deps.py, so configuring needs no paths.
# The folder layout and defaults match the script.

# Default install folder: TENGEN_DEPS_DIR environment variable, else the user's data folder.
function(tengen_default_deps_dir result)
	if(DEFINED ENV{TENGEN_DEPS_DIR})
		set(dir "$ENV{TENGEN_DEPS_DIR}")
	elseif(WIN32)
		set(dir "$ENV{LOCALAPPDATA}/tengen/deps")
	elseif(DEFINED ENV{XDG_DATA_HOME})
		set(dir "$ENV{XDG_DATA_HOME}/tengen/deps")
	else()
		set(dir "$ENV{HOME}/.local/share/tengen/deps")
	endif()
	file(TO_CMAKE_PATH "${dir}" dir)
	set(${result} "${dir}" PARENT_SCOPE)
endfunction()

# Qt prefix: <deps>/qt/<arch>/<version>/<folder>. aqt names the last folder itself (e.g. msvc2022_64), so look it up.
function(tengen_qt_prefix result depsDir arch version)
	file(GLOB qtLib "${depsDir}/qt/${arch}/${version}/*/lib")
	get_filename_component(prefix "${qtLib}" DIRECTORY)
	set(${result} "${prefix}" PARENT_SCOPE)
endfunction()

# OpenCV prefix: <deps>/opencv-<version>. The official Windows package nests it in opencv/build.
function(tengen_opencv_prefix result depsDir version)
	set(prefix "${depsDir}/opencv-${version}")
	if(WIN32)
		string(APPEND prefix "/opencv/build")
	endif()
	set(${result} "${prefix}" PARENT_SCOPE)
endfunction()


# ----- Settings -----

if(WIN32)
	set(defaultQtArch win64_msvc2022_64)
else()
	set(defaultQtArch linux_gcc_64)
endif()
tengen_default_deps_dir(defaultDepsDir)

set(TENGEN_DEPS_DIR "${defaultDepsDir}" CACHE PATH   "Install folder of scripts/setup_deps.py.")
set(TENGEN_QT_ARCH  "${defaultQtArch}"  CACHE STRING "Qt architecture, see scripts/setup_deps.py --qt-arch.")


# ----- Search paths -----

tengen_qt_prefix(qtPrefix "${TENGEN_DEPS_DIR}" "${TENGEN_QT_ARCH}" "${TENGEN_QT_VERSION}")
tengen_opencv_prefix(opencvPrefix "${TENGEN_DEPS_DIR}" "${TENGEN_OPENCV_VERSION}")

# Included from lib/CMakeLists.txt: PARENT_SCOPE makes the paths visible to the whole project, not only lib/.
list(APPEND CMAKE_PREFIX_PATH ${qtPrefix} "${opencvPrefix}") # qtPrefix unquoted: adds nothing if Qt is not installed.
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" PARENT_SCOPE)
