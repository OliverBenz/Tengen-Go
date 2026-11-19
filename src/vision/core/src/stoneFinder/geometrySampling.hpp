#pragma once

#include "stoneFinderTypes.hpp"

namespace tengen::vision::core {
namespace GeometrySampling {

Radii chooseRadii(double spacing, const GeometryConfig& config);
std::vector<cv::Point> makeCircleOffsets(int radius);
Offsets precomputeOffsets(const Radii& radii);

} // namespace GeometrySampling
} // namespace tengen::vision::core
