#pragma once

#include <array>
#include <filesystem>
#include <opencv2/core/types.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tengen::vision::analysis {

//! Four corners of an outline in original image space, stored unordered (see resources/README.md).
using Corners = std::array<cv::Point2f, 4>;

//! Manually labelled geometry for one image.
struct GroundTruth {
	unsigned boardSize{};
	Corners board; //!< Outer edge of the physical board.
	Corners grid;  //!< Outermost grid-line intersections.
};

//! One directory of test images.
struct ImageSet {
	std::string name;
	std::vector<std::filesystem::path> images; //!< Naturally ordered, so reports diff cleanly.
	bool labelled{};                           //!< Images have .json ground truth next to them.
};

//! Collect every directory under \p root that holds images.
//! Sets without ground truth are treated as fixed-camera series and measured against their own median.
std::vector<ImageSet> discoverImageSets(const std::filesystem::path& root);

std::optional<GroundTruth> loadGroundTruth(const std::filesystem::path& imagePath);

} // namespace tengen::vision::analysis
