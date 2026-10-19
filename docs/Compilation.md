# Compilation Instructions

The project is configured via [`CMakePresets.json`](../CMakePresets.json): `Linux64-Debug`/`Linux64-Release`/`Linux64-RelWithDebInfo` (Ninja) on Linux, and `Win64` (Visual Studio 17 2022, Debug/Release/RelWithDebInfo) on Windows.

Besides the dependencies CMake fetches itself, you need Qt 6.8 or newer and OpenCV 5.

## Linux

Install Qt 6 and OpenCV through your package manager, e.g. on Arch:

```
sudo pacman -S qt6-base qt6-multimedia opencv
```

Then configure and build:

```
cmake --preset Linux64-Debug && cmake --build --preset Linux64-Debug
```

## Windows

You need Visual Studio 2022 with the C++ workload, CMake 3.27 or newer, and Python 3.

[`scripts/win/1_setup_deps.py`](../scripts/win/1_setup_deps.py) installs the Qt and OpenCV versions Tengen is tested with into `%LOCALAPPDATA%\tengen\deps`.
CMake finds them there by itself, and the build copies the DLLs next to the executables, so nothing goes into `PATH`.

```
python scripts/win/1_setup_deps.py
cmake --preset Win64 && cmake --build --preset Win64-Debug
```

`--deps-dir` (or the `TENGEN_DEPS_DIR` environment variable) installs somewhere else, `--help` lists all options.
