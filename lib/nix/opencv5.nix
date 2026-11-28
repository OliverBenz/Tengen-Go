# OpenCV 5, which nixpkgs does not package yet.
# Only the modules Tengen uses, so the source build stays short. BUILD_LIST adds their dependencies (e.g. flann).
{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  pkg-config,
  zlib,
  libpng,
  libjpeg,
  libtiff,
  libwebp,
  gtk3,
  ffmpeg-headless,
}:

stdenv.mkDerivation (finalAttrs: {
  pname = "opencv";
  version = "5.0.0";

  src = fetchFromGitHub {
    owner = "opencv";
    repo = "opencv";
    tag = finalAttrs.version;
    hash = "sha256-fmMHLQ5IHDKztuIaetQj/deCl82pVYh+Dvp8G9Xb5bk=";
  };

  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
  ];

  buildInputs = [
    zlib
    libpng
    libjpeg
    libtiff
    libwebp
    gtk3 # highgui backend (imshow in the debug visualizer).
    ffmpeg-headless # videoio for video files. Cameras use V4L, which needs no extra package.
  ];

  cmakeFlags = [
    (lib.cmakeFeature "CMAKE_BUILD_TYPE" "Release")
    (lib.cmakeFeature "BUILD_LIST" "core,imgproc,imgcodecs,videoio,highgui,geometry,dnn")

    # Nothing but the libraries.
    (lib.cmakeBool "BUILD_TESTS" false)
    (lib.cmakeBool "BUILD_PERF_TESTS" false)
    (lib.cmakeBool "BUILD_EXAMPLES" false)
    (lib.cmakeBool "BUILD_DOCS" false)
    (lib.cmakeBool "BUILD_opencv_apps" false)
    (lib.cmakeBool "BUILD_JAVA" false)
    (lib.cmakeBool "BUILD_opencv_python3" false)

    # These download at configure time, which the Nix sandbox forbids.
    (lib.cmakeBool "WITH_IPP" false)
    (lib.cmakeBool "WITH_UNIFONT" false)
    (lib.cmakeBool "WITH_ADE" false)

    # Use the nixpkgs libraries instead of the bundled copies.
    (lib.cmakeBool "BUILD_ZLIB" false)
    (lib.cmakeBool "BUILD_PNG" false)
    (lib.cmakeBool "BUILD_JPEG" false)
    (lib.cmakeBool "BUILD_TIFF" false)
    (lib.cmakeBool "BUILD_WEBP" false)

    # Bundled protobuf for dnn's ONNX import.
    (lib.cmakeBool "WITH_PROTOBUF" true)
    (lib.cmakeBool "BUILD_PROTOBUF" true)

    (lib.cmakeBool "WITH_GTK" true)
    (lib.cmakeBool "WITH_QT" false)
    (lib.cmakeBool "WITH_FFMPEG" true)
    (lib.cmakeBool "WITH_V4L" true)
    (lib.cmakeBool "WITH_GSTREAMER" false)
    (lib.cmakeBool "WITH_1394" false)
    (lib.cmakeBool "WITH_CUDA" false)
    (lib.cmakeBool "WITH_OPENCL" false)
    (lib.cmakeBool "WITH_OPENEXR" false)
    (lib.cmakeBool "WITH_OPENJPEG" false)
    (lib.cmakeBool "WITH_JASPER" false)
    (lib.cmakeBool "WITH_EIGEN" false)

    # CMake finds OpenCVConfig.cmake. The generated .pc file joins the absolute Nix paths wrongly (prefix//nix/store/...).
    (lib.cmakeBool "OPENCV_GENERATE_PKGCONFIG" false)
  ];

  meta = {
    description = "Open Computer Vision Library, reduced to the modules Tengen uses";
    homepage = "https://opencv.org";
    license = lib.licenses.asl20;
    platforms = lib.platforms.linux;
  };
})
