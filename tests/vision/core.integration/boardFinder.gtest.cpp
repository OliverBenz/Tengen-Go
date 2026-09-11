#include "vision/core/gridFinder.hpp"

#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <vector>

// The boardFinder tests only verify that, given an image of a go board, we can detect the board in the image.
// This is done using the .json files next to each image containing the board and grid coordiantes.
namespace tengen::vision::core {
namespace gtest {

//! Min-dimension of the axis-aligned bounding box of 4 points, in image-space pixels.
static float boundingBoxMinDim(const std::array<cv::Point2f, 4>& points) {
	float minX = points[0].x, maxX = points[0].x;
	float minY = points[0].y, maxY = points[0].y;
	for (const auto& p: points) {
		minX = std::min(minX, p.x);
		maxX = std::max(maxX, p.x);
		minY = std::min(minY, p.y);
		maxY = std::max(maxY, p.y);
	}
	return std::min(maxX - minX, maxY - minY);
}


//! Load the png files in the given resource subdirectory.
void runTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Get test images and ensure valid
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	static constexpr float TOLERANCE_FRACTION = 0.1f; //!< Percentage of acceptable pixel position error relative to the contour bounding box.

	for (const auto& imagePath: images) {
		// Load image
		cv::Mat image = cv::imread(imagePath.string());
		ASSERT_FALSE(image.empty());

		// Find the Go board
		const auto warpResult = warpToBoard(image);
		EXPECT_TRUE(isValidBoard(warpResult));

		// Load the real geometry from the json file
		const auto jsonPath          = std::filesystem::path(imagePath).replace_extension(".json");
		GeometryGroundTruth geometry = GeometryGroundTruth::loadFromFile(jsonPath);

		// Check the found contour corners align with the ones defined in the json file
		// It may be the case that we find the grid contour instead of the board contour. This is suboptimal but not wrong. We output and continue
		const float boardTolerance = TOLERANCE_FRACTION * boundingBoxMinDim(geometry.boardCorners);
		if (!pointSetsMatch(warpResult.contourCorners, geometry.boardCorners, boardTolerance)) {
			std::cout << std::format("Board Corners not matched in '{}'!\n", imagePath.string());

			const float gridTolerance = TOLERANCE_FRACTION * boundingBoxMinDim(geometry.gridCorners);
			if (!pointSetsMatch(warpResult.contourCorners, geometry.gridCorners, gridTolerance)) {
				for (std::size_t i = 0; i < 4; ++i)
					std::cout << "  contourCorners[" << i << "]=" << warpResult.contourCorners[i] << '\n';
				for (std::size_t i = 0; i < 4; ++i)
					std::cout << "  boardCorners[" << i << "]=" << geometry.boardCorners[i] << '\n';
				for (std::size_t i = 0; i < 4; ++i)
					std::cout << "  gridCorners[" << i << "]=" << geometry.gridCorners[i] << '\n';
				EXPECT_TRUE(false);
			}
		}
	}
}

TEST(BoardFinder, Angled_Easy) {
	runTest("angled_easy", 6u);
}

TEST(BoardFinder, Angled_Hard) {
	runTest("angled_hard", 6u);
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
