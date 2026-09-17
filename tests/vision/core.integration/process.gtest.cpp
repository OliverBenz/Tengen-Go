#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <string>

// The process tests verify the whole vision pipeline at once: board detection, grid detection and stone detection.
// Nothing is prepared, so an image enters the pipeline exactly as the application would hand it over. The outcome is
// checked against the labelled geometry (.json) and the stone layout (.txt) of each image.
// These test sets require a triple: (<image>.jpeg, <image>.json, <layout>.txt) as documented in resources/README.md.
namespace tengen::vision::core {
namespace gtest {

static constexpr float TOLERANCE_FRACTION = 0.1f; //!< Percentage of acceptable error relative to the contour bounding box / spacing.

//! Runs the full pipeline on every image of the test set and verifies the detected board against the defined ground truth.
static void runTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Get test images and ensure valid
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	// Checks below are non-fatal so one bad image does not hide the results of the remaining ones.
	for (const auto& imagePath: images) {
		// Load image
		const cv::Mat image = cv::imread(imagePath.string());
		if (image.empty()) {
			ADD_FAILURE() << "Could not load " << imagePath << "\n";
			continue;
		}

		const PipelineResult result = runPipeline(image);
		if (!isValidPipelineResult(result)) {
			ADD_FAILURE() << "Pipeline produced no usable result for " << imagePath << "\n";
			continue;
		}

		// Load geometry information
		const auto jsonPath                = std::filesystem::path(imagePath).replace_extension(".json");
		const GeometryGroundTruth geometry = GeometryGroundTruth::loadFromFile(jsonPath);

		// The detected board must match the labelled geometry.
		EXPECT_TRUE(boardContourMatchesEitherOutline(result.warped.contourCorners, geometry, TOLERANCE_FRACTION))
		        << "BoardFinder contour matches neither the board nor the grid outline for " << imagePath;
		verifyRectifiedBoard(result.rectified, geometry);

		// The detected stones are indexed by the detected grid, so a wrong board size makes the comparison meaningless.
		if (result.rectified.geometry.boardSize != geometry.boardSize) {
			continue;
		}
		expectStonesMatchBoard(result.stoneStep.stones, loadExpectedBoard(imagePath), imagePath.string());
	}
}


TEST(Process, Empty_Angle_None) {
	runTest("empty_angle_none", 3u);
}

TEST(Process, Empty_Angle_Small) {
	runTest("empty_angle_small", 3u);
}

// TODO: The stone detection fails on angle_3 and angle_4 (see StoneFinder.DISABLED_Ideal_Angled_Easy).
TEST(Process, DISABLED_Angled_Easy) {
	runTest("angled_easy", 6u);
}

// TODO: The pipeline does not survive the lighting of this test set yet (see GridFinder.DISABLED_Full_Angled_Hard_Lighting).
TEST(Process, DISABLED_Angled_Hard_Lighting) {
	runTest("angled_hard_lighting", 6u);
}

// TODO: This test set has no stone layout (board.txt) yet.
TEST(Process, DISABLED_Angled_Hard) {
	runTest("angled_hard", 8u);
}

// TODO: The game series have no labelled geometry (.json) yet.
TEST(Process, DISABLED_Game_Simple_Size9) {
	runTest("game_simple/size_9", 14u);
}

// TODO: The game series have no labelled geometry (.json) yet. Once enabled, move_24 misses the stone at (12, 8).
TEST(Process, DISABLED_Game_Simple_Size13) {
	runTest("game_simple/size_13", 28u);
}

} // namespace gtest
} // namespace tengen::vision::core
