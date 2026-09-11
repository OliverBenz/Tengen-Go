#pragma once

#include "vision/core/gridFinder.hpp"

#include <filesystem>
#include <opencv2/opencv.hpp>

namespace tengen::vision::core {
namespace gtest {

//! Manually labeled board geometry. All points are in the original image space.
struct GeometryGroundTruth {
	unsigned boardSize{};
	std::array<cv::Point2f, 4> boardCorners; //!< Outer edge of the physical board (4 points, unordered).
	std::array<cv::Point2f, 4> gridCorners;  //!< Outermost grid-line intersections (4 points, unordered).

	static GeometryGroundTruth loadFromFile(const std::filesystem::path& jsonPath);
};


//! Check if two sets of points match given some tolerance. Takes into account that points arrays may be permuted.
bool pointSetsMatch(const std::array<cv::Point2f, 4>& expected, const std::array<cv::Point2f, 4>& actual, float tolerance);

//! Verify the board geometry resulting from the vision algorithm matches the test defined board geometry.
void verifyBoardGeometry(const BoardGeometry& result, const GeometryGroundTruth& geometry, float pointDeviationPercentage = 0.1f);

//! Verify the rectified board data from the vision algorithm matches the test defined board geometry.
void verifyRectifiedBoard(const RectifiedBoard& board, const GeometryGroundTruth& geometry);

//! Get the minimum distance between four corner points.
float minimumCornerPointDistance(const std::array<cv::Point2f, 4>& points);

} // namespace gtest
} // namespace tengen::vision::core
