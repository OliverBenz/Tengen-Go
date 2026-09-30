# Compilation Instructions

The project is configured via [`CMakePresets.json`](../CMakePresets.json): `Win64` (Visual Studio 17 2022, Debug/Release/RelWithDebInfo) on Windows, and `Linux64-Debug`/`Linux64-Release`/`Linux64-RelWithDebInfo` (Ninja) on Linux.

Besides the dependencies CMake fetches itself, you need Qt 6 and OpenCV.

## Linux

Install Qt 6 and OpenCV through your package manager, then configure and build:

```
cmake --preset Linux64-Debug && cmake --build --preset Linux64-Debug
```

## Windows

Windows has no package manager that CMake picks these up from, so Qt and OpenCV are installed by hand and made known through environment variables.
You also need Visual Studio 2022 with the C++ workload and CMake 3.27 or newer.

### OpenCV

1. Download the Windows installer from the [OpenCV releases](https://github.com/opencv/opencv/releases) on GitHub and extract it, e.g. to `C:\opencv`.
2. Add the folder holding the OpenCV DLLs to `PATH`, e.g. `C:\opencv\build\x64\vc16\bin`.

CMake finds OpenCV through `OpenCV_DIR`, the folder containing `OpenCVConfig.cmake`.
The `Win64` preset sets it to `C:/opencv/build`; if your install lives elsewhere, edit that path or override it in a local `CMakeUserPresets.json`.

### Qt

1. Get Qt 6:
   - With a Qt account, use the [Qt online installer](https://www.qt.io/download-qt-installer) and pick the MSVC 2022 64-bit build.
   - Without one, [build Qt from source](https://doc.qt.io/qt-6/windows-building.html).
2. Add Qt's `bin` folder to `PATH`, e.g. `C:\Qt\6.8.0\msvc2022_64\bin`.

CMake finds Qt through that `PATH` entry, and Tengen finds the Qt DLLs through it at runtime.

### Build

Open a new terminal so it picks up the changed `PATH`, then:

```
cmake --preset Win64 && cmake --build --preset Win64-Debug
```
