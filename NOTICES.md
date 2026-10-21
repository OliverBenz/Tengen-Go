# Notices

Which license covers which part of Tengen and of what ships with it.
The license texts are in the same folder as this file, third-party texts in its `3rdparty/` subfolder. In the repository, they are in `LICENSES/` (Tengen's own) and `3rdparty/licenses/` (third parties).

Source code: <https://github.com/OliverBenz/Tengen-Go>. The tag of a release holds the source of exactly that release.

## Tengen

| Part                                         | License       | Copyright        | License text        |
| -------------------------------------------- | ------------- | ---------------- | ------------------- |
| Tengen, everything not listed below          | AGPL-3.0-only | 2024 Oliver Benz | `AGPL-3.0-only.txt` |
| netCore (networking, part of Tengen)         | Apache-2.0    | 2024 Oliver Benz | `Apache-2.0.txt`    |
| visionCore (board detection, part of Tengen) | Apache-2.0    | 2024 Oliver Benz | `Apache-2.0.txt`    |
| visionPerception (board perception)          | AGPL-3.0-only | 2024 Oliver Benz | `AGPL-3.0-only.txt` |
| Stone sounds (`resources/audio`)             | CC0-1.0       | 2024 Oliver Benz | `CC0-1.0.txt`       |

The license of every file in the repository is listed in `REUSE.toml`.

## Resources

Shipped in `resources/`. Its README lists each file with its source.

| Part                                      | License    | Copyright                | License text     |
| ----------------------------------------- | ---------- | ------------------------ | ---------------- |
| Stones and board of the OGS "Anime" theme | Apache-2.0 | Online-Go.com            | `Apache-2.0.txt` |
| Board textures                            | CC0-1.0    | ShareTextures, ambientCG | `CC0-1.0.txt`    |

## Libraries built into Tengen

| Library               | Version        | License | Copyright                         | License text                             |
| --------------------- | -------------- | ------- | --------------------------------- | ---------------------------------------- |
| [nlohmann/json][json] | 3.12.0         | MIT     | 2013-2025 Niels Lohmann           | `3rdparty/LICENSE-MIT-nlohmann_json.txt` |
| [Logger][logger]      | commit 8e8f8e4 | MIT     | 2021 Oliver Benz                  | `3rdparty/LICENSE-MIT-Logger.txt`        |
| [Asio][asio]          | 1.36.0         | BSL-1.0 | 2003-2025 Christopher M. Kohlhoff | `3rdparty/LICENSE-BSL-1.0.txt`           |

## Libraries shipped next to Tengen

Only in the Windows package. On Linux these come from the system.

| Library                                                                               | Version                                   | License                   | Copyright                                  | License text                                                    |
| ------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------- | ------------------------------------------ | --------------------------------------------------------------- |
| [Qt][qt]: Core, Gui, Widgets, Concurrent, Multimedia, Network, Svg, and their plugins | 6.12.0                                    | LGPL-3.0-only             | The Qt Company Ltd. and other contributors | `3rdparty/LICENSE-LGPL-3.0.txt`, `3rdparty/LICENSE-GPL-3.0.txt` |
| [FFmpeg][ffmpeg], as built by Qt for Qt Multimedia                                    | 9.0.1                                     | LGPL-2.1-or-later         | The FFmpeg developers                      | `3rdparty/LICENSE-LGPL-2.1.txt`                                 |
| [Mesa][mesa] llvmpipe, software OpenGL (`opengl32sw.dll`), as built by Qt             | 11.2.2                                    | MIT, BSL-1.0              | 1999-2007 Brian Paul, 2013-2016 The Khronos Group Inc. (MIT); 2012 yohhoy (BSL-1.0, C11 threads emulation) | `3rdparty/LICENSE-MIT-Mesa.txt`, `3rdparty/LICENSE-BSL-1.0.txt` |
| [LLVM][llvm], inside `opengl32sw.dll`                                                 | as built by Qt                            | Apache-2.0 WITH LLVM-exception | LLVM Team                             | `3rdparty/LICENSE-Apache-2.0-LLVM.txt`                          |
| [OpenCV][opencv]: `opencv_world`                                                      | 5.0.0                                     | Apache-2.0                | Intel Corporation, Willow Garage, NVIDIA, AMD, OpenCV Foundation, Itseez, Xperience AI, OpenCV AI, and other contributors | `Apache-2.0.txt`                       |
| Microsoft Visual C++ runtime, `D3DCompiler_47.dll`                                    |                                           | Microsoft redistributable | Microsoft Corporation                      |                                                                 |

Qt itself contains third-party code under further licenses, listed in [Licenses Used in Qt][qt-licenses].
OpenCV contains third-party libraries under their own licenses (e.g. image codecs and Intel IPP). Their texts are in `3rdparty/opencv/`, copied from OpenCV's package.
OpenCV's FFmpeg video plugin (`opencv_videoio_ffmpeg*.dll`) contains FFmpeg under LGPL-2.1-or-later, see `3rdparty/opencv/`.

Qt and FFmpeg are linked dynamically. You may replace their DLLs next to `tengen.exe` with your own builds of the same libraries.
Their source code:

- Qt 6.12.0: <https://download.qt.io/official_releases/qt/6.12/6.12.0/single/>
- FFmpeg 9.0.1: <https://ffmpeg.org/releases/ffmpeg-9.0.1.tar.gz>, built with Qt's scripts in <https://code.qt.io/cgit/qt/qt5.git/tree/coin/provisioning>
- The same sources are attached to each Tengen release on GitHub.

[json]: https://github.com/nlohmann/json
[logger]: https://github.com/OliverBenz/Logger
[asio]: https://think-async.com/Asio/
[qt]: https://www.qt.io
[qt-licenses]: https://doc.qt.io/qt-6/licenses-used-in-qt.html
[ffmpeg]: https://ffmpeg.org
[mesa]: https://www.mesa3d.org
[llvm]: https://llvm.org
[opencv]: https://opencv.org
