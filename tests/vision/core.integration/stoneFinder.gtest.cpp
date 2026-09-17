#include "vision/core/stoneFinder.hpp"

#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <utility>

// The stoneFinder tests only verify that, given an image of a go board, we can detect the stones lying on it.
// The board corners defined in the .json file next to each image are used to warp an ideal board image, the same way
// the gridFinder tests do. The stone detection therefore works on a perfect input and a failure here is a stone
// detection problem rather than a board or grid detection problem.
// These test sets require a triple: (<image>.jpeg, <image>.json, <layout>.txt) as documented in resources/README.md.
namespace tengen::vision::core {
namespace gtest {

static constexpr float TOLERANCE_FRACTION = 0.1f; //!< Percentage of acceptable error relative to the contour bounding box.

//! Given an input image, prepares an ideal board image based on the defined board corners, then tests the stone detection with this prepared image.
static void runTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Get test images and ensure valid
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	// Checks below are non-fatal so one bad image does not hide the results of the remaining ones.
	for (const auto& imagePath: images) {
		// A) Prepare the test
		// Load image
		const cv::Mat image = cv::imread(imagePath.string());
		if (image.empty()) {
			ADD_FAILURE() << "Could not load " << imagePath << "\n";
			continue;
		}

		// Load geometry information
		const auto jsonPath                = std::filesystem::path(imagePath).replace_extension(".json");
		const GeometryGroundTruth geometry = GeometryGroundTruth::loadFromFile(jsonPath);
		const float boardTolerance         = TOLERANCE_FRACTION * minimumCornerPointDistance(geometry.boardCorners);

		// Warp the image onto the defined board corners to get an ideal boardFinder output.
		const WarpResult warpResult = prepareIdealImage(image, geometry.boardCorners);
		if (!pointSetsMatch(warpResult.contourCorners, geometry.boardCorners, boardTolerance)) {
			ADD_FAILURE() << "Ideal warp did not reproduce the labelled board corners for " << imagePath << "\n";
			continue;
		}

		// B) Start of the stone detection test
		const PipelineResult result = runPipeline(image, warpResult);
		if (!isValidPipelineResult(result)) {
			ADD_FAILURE() << "Stone detection produced no usable result for " << imagePath << "\n";
			continue;
		}

		// The detected stones are indexed by the detected grid, so a wrong board size makes the comparison meaningless.
		if (result.rectified.geometry.boardSize != geometry.boardSize) {
			ADD_FAILURE() << "Detected board size " << result.rectified.geometry.boardSize << " instead of " << geometry.boardSize << " for " << imagePath
			              << "\n";
			continue;
		}

		expectStonesMatchBoard(result.stoneStep.stones, loadExpectedBoard(imagePath), imagePath.string());
	}
}


//! The single stone lying on the board, or Empty if the board does not hold exactly one stone.
static StoneState singleStoneOf(const StoneResult& result) {
	const auto isStone = [](const StoneState state) { return state == StoneState::Black || state == StoneState::White; };

	const auto stone = std::find_if(result.stones.begin(), result.stones.end(), isStone);
	if (stone == result.stones.end() || std::any_of(std::next(stone), result.stones.end(), isStone)) {
		return StoneState::Empty;
	}
	return *stone;
}

//! The setup images are already top-down views, so fall back to the unchanged image if no board contour is found.
static WarpResult warpSetupImage(const cv::Mat& image) {
	const WarpResult warpResult = warpToBoard(image);
	if (isValidBoard(warpResult)) {
		return warpResult;
	}
	return {image, cv::Mat::eye(3, 3, CV_64F), {}};
}


TEST(StoneFinder, Ideal_Empty_Angle_None) {
	runTest("empty_angle_none", 3u);
}

TEST(StoneFinder, Ideal_Empty_Angle_Small) {
	runTest("empty_angle_small", 3u);
}

// TODO: angle_3 and angle_4 show the board from the far side. Two of their white stones are missed and two empty intersections are reported white.
TEST(StoneFinder, DISABLED_Ideal_Angled_Easy) {
	runTest("angled_easy", 6u);
}

// TODO: angle_3 and angle_5 each miss a single white stone, angle_4 is off by 14 intersections.
TEST(StoneFinder, DISABLED_Ideal_Angled_Hard_Lighting) {
	runTest("angled_hard_lighting", 6u);
}

// TODO: This test set has no stone layout (board.txt) yet.
TEST(StoneFinder, DISABLED_Ideal_Angled_Hard) {
	runTest("angled_hard", 8u);
}

// The setup images show a single stone on an empty board and are used to set up the perception algorithm.
// We only check here that the gauge stone is detected. The more complex checks on these images are in the perception tests.
TEST(StoneFinder, Setup_SingleStone) {
	static constexpr std::array<std::pair<std::string_view, StoneState>, 12> CASES = {{
	        {"C2_1.png", StoneState::Black},
	        {"C2_2.png", StoneState::Black},
	        {"C2_3.png", StoneState::Black},
	        {"C2_4.png", StoneState::Black},
	        {"E3_1.png", StoneState::Black},
	        {"E3_2.png", StoneState::Black},
	        {"E3_3.png", StoneState::Black},
	        {"E3_4.png", StoneState::Black},
	        {"C2_1_white.png", StoneState::White},
	        {"C2_2_white.png", StoneState::White},
	        {"C2_3_white.png", StoneState::White},
	        {"C2_4_white.png", StoneState::White},
	}};

	for (const auto& [fileName, expectedState]: CASES) {
		const auto imagePath = std::filesystem::path(PATH_TEST_IMG) / "setup" / std::string(fileName);

		const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
		if (image.empty()) {
			ADD_FAILURE() << "Could not load " << imagePath << "\n";
			continue;
		}

		const PipelineResult result = runPipeline(image, warpSetupImage(image));
		if (!isValidPipelineResult(result)) {
			ADD_FAILURE() << "Stone detection produced no usable result for " << imagePath << "\n";
			continue;
		}

		EXPECT_EQ(singleStoneOf(result.stoneStep), expectedState) << imagePath;
	}
}

} // namespace gtest
} // namespace tengen::vision::core
