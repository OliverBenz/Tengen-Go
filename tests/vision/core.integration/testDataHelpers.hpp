#pragma once

#include "model/board.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

namespace tengen::vision::core {
namespace gtest {

//! Expected test results (result of each step in the pipeline).
struct TestResult {
	WarpResult warped;
	RectifiedBoard rectified;
	StoneResult stoneStep;
};

//! Run the stone detection pipeline. Ensure intermediate steps are generally valid. Return test result.
TestResult runPipeline(const std::filesystem::path& imgPath);

//! Load the dotBW ground truth board matching an image path (same file name, ".txt" extension).
Board loadExpectedBoard(const std::filesystem::path& imagePath);

//! Check every board coordinate  against a ground truth board.
void expectStonesMatchBoard(const std::vector<StoneState>& stones, unsigned boardSize, const Board& expected);


//! Match two equally-sized point sets without assuming a fixed order (a board photographed at a
//! strong angle has no well-defined "top-left" corner), then check every matched pair is within
//! tolerance. Brute-forces all permutations, which is fine for the small (<=4) point sets we use this for.
void expectPointsMatch(const std::vector<cv::Point2f>& expected, const std::vector<cv::Point2f>& actual, float tolerance, std::string_view context);

//! Non-asserting counterpart of expectPointsMatch(): the largest per-point distance under the same
//! best-match permutation. Lets a caller report the error continuously instead of as pass/fail.
//! \returns Worst matched distance in pixels, or infinity if the sets are empty or differently sized.
double maxMatchedPointDistance(const std::vector<cv::Point2f>& expected, const std::vector<cv::Point2f>& actual);

//! Intersection-over-union of two quads. Both are convex-hulled first, so corner order does not matter
//! (ground truth corners are stored unordered, see resources/README.md).
//! \returns IoU in [0, 1], or 0.0 if either quad is degenerate.
double quadIoU(const std::vector<cv::Point2f>& lhs, const std::array<cv::Point2f, 4>& rhs);

} // namespace gtest
} // namespace tengen::vision::core
