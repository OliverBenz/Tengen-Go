#pragma once

#include <span>

namespace tengen::vision::core {

//! stoneNet.onnx, compiled in by cmake/embedFile.cmake.
std::span<const unsigned char> stoneNetModel();

} // namespace tengen::vision::core
