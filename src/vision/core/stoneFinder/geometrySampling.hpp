#pragma once

#include "stoneFinderInternal.hpp"

namespace tengen::vision::core::GeometrySampling {

Radii chooseRadii(double spacing, const GeometryConfig& config);
std::vector<cv::Point> makeCircleOffsets(int radius);
Offsets precomputeOffsets(const Radii& radii);

} // namespace tengen::vision::core::GeometrySampling
