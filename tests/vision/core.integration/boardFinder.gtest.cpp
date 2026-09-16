#include "vision/core/gridFinder.hpp"

#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <array>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <string>

// The boardFinder tests only verify that, given an image of a go board, we can detect the board in the image.
// This is done using the .json files next to each image containing the board and grid coordiantes.
namespace tengen::vision::core {
namespace gtest {

//! Format a corner set for a failure message.
static std::string formatCorners(const std::array<cv::Point2f, 4>& points) {
	std::string text;
	for (const auto& p: points) {
		text += std::format(" ({:.1f}, {:.1f})", p.x, p.y);
	}
	return text;
}


//! Load the png files in the given resource subdirectory.
void runTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Get test images and ensure valid
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	static constexpr float TOLERANCE_FRACTION = 0.1f; //!< Percentage of acceptable pixel position error relative to the contour bounding box.

	// NOTE: Every per-image check below is non-fatal and skips to the next image, so one bad image reports
	//       itself instead of aborting the sweep and hiding whether the remaining images pass.
	for (const auto& imagePath: images) {
		// Load image
		cv::Mat image = cv::imread(imagePath.string());
		if (image.empty()) {
			ADD_FAILURE() << "Could not load " << imagePath << "\n";
			continue;
		}

		// Find the Go board
		const auto warpResult = warpToBoard(image);
		if (!isValidBoard(warpResult)) {
			ADD_FAILURE() << "BoardFinder produced no valid board for " << imagePath << "\n";
			continue;
		}

		// Load the real geometry from the json file
		const auto jsonPath          = std::filesystem::path(imagePath).replace_extension(".json");
		GeometryGroundTruth geometry = GeometryGroundTruth::loadFromFile(jsonPath);

		// Check that we found either the board contour or the grid contour.
		if (!boardContourMatchesEitherOutline(warpResult.contourCorners, geometry, TOLERANCE_FRACTION)) {
			ADD_FAILURE() << "Contour matches neither the board nor the grid outline in '" << imagePath.string() << "'.\n"
			              << "  contour:" << formatCorners(warpResult.contourCorners) << "\n"
			              << "  board:  " << formatCorners(geometry.boardCorners) << "\n"
			              << "  grid:   " << formatCorners(geometry.gridCorners);
		}
	}
}

TEST(BoardFinder, Angled_Easy) {
	runTest("angled_easy", 6u);
}

TEST(BoardFinder, Angled_Hard) {
	runTest("angled_hard", 8u);
}

TEST(BoardFinder, DISABLED_Angled_Hard_Lighting) {
	runTest("angled_hard_lighting", 6u);
}

TEST(BoardFinder, Empty_Angle_None) {
	runTest("empty_angle_none", 3u);
}

TEST(BoardFinder, Empty_Angle_Small) {
	runTest("empty_angle_small", 3u);
}

// TODO: Test the two games. These do not have json files yet.

} // namespace gtest
} // namespace tengen::vision::core
