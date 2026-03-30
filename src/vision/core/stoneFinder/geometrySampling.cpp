#include "geometrySampling.hpp"

#include "stoneFinderInternal.hpp"

#include <algorithm>
#include <cmath>

namespace tengen::vision::core {
namespace GeometrySampling {

Radii chooseRadii(double spacing, const GeometryConfig& config) {
	Radii radii{};
	radii.innerRadius = roundedSpacingValue(spacing, config.innerRadiusSpacingK, config.innerRadiusFallback, config.innerRadiusMin, config.innerRadiusMax);
	radii.bgRadius    = std::clamp(radii.innerRadius / 2, config.bgRadiusMin, config.bgRadiusMax);

	if (std::isfinite(spacing) && spacing > 0.0) {
		const int bgOffset = static_cast<int>(std::lround(spacing * config.bgOffsetSpacingK));
		radii.bgOffset     = std::max(bgOffset, radii.innerRadius + config.bgOffsetMinExtra);
	} else {
		radii.bgOffset = radii.innerRadius * 2 + config.bgOffsetFallbackAdd;
	}
	return radii;
}

std::vector<cv::Point> makeCircleOffsets(int radius) {
	std::vector<cv::Point> offsets;
	offsets.reserve(static_cast<std::size_t>((2 * radius + 1) * (2 * radius + 1)));
	const int radiusSquared = radius * radius;
	for (int deltaY = -radius; deltaY <= radius; ++deltaY) {
		for (int deltaX = -radius; deltaX <= radius; ++deltaX) {
			if (deltaX * deltaX + deltaY * deltaY <= radiusSquared) {
				offsets.emplace_back(deltaX, deltaY);
			}
		}
	}
	return offsets;
}

Offsets precomputeOffsets(const Radii& radii) {
	Offsets offsets{};
	offsets.inner = makeCircleOffsets(radii.innerRadius);
	offsets.bg    = (radii.bgRadius == radii.innerRadius) ? offsets.inner : makeCircleOffsets(radii.bgRadius);
	return offsets;
}

} // namespace GeometrySampling
} // namespace tengen::vision::core
