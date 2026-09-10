#pragma once

#include <filesystem>
#include <opencv2/opencv.hpp>

namespace tengen::vision::core {
namespace gtest {

//! Manually labeled board geometry. All points are in the original image space.
struct GeometryGroundTruth {
	unsigned boardSize{};
	std::array<cv::Point2f, 4> boardCorners; //!< Outer edge of the physical board (4 points, unordered).
	std::array<cv::Point2f, 4> gridCorners;  //!< Outermost grid-line intersections (4 points, unordered).
};

//! Load the geometry ground truth matching an image path (same file name, ".json" extension).
GeometryGroundTruth loadGeometryGroundTruth(const std::filesystem::path& imagePath);

//! Check if two sets of points match given some tolerance. Takes into account that points arrays may be permuted.
bool pointSetsMatch(const std::array<cv::Point2f, 4>& expected, const std::array<cv::Point2f, 4>& actual, float tolerance);

} // namespace gtest
} // namespace tengen::vision::core
