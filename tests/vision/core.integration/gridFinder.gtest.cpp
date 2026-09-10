#include "vision/core/gridFinder.hpp"

#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/core/types.hpp>

// TODO: Can we perform more tests on the transformed image?
// TODO: The two runTest functions only differ by how we WarpToBoard. Could merge the functions.

// The gridFinder tests only verify that, given an image of a go board, we can detect the grid in the image.
// This is done using the .json files next to each image containing the board and grid coordiantes in two ways:
//  1) We use the defined board corner coordiantes in the json files to produce an ideal boardFinder output.
//  2) We use the actual output of the boardFinder step.
// These tests require a pair: (<image>.jpg, <image>.json) where the json describes the geometry as documented.
namespace tengen::vision::core {
namespace gtest {

static constexpr float TOLERANCE_FRACTION = 0.1f; //!< Percentage of acceptable error relative to the contour bounding box / spacing.

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

//! Given the original image and the four defined corner points of the board, computes the boardFinder transformation.
WarpResult prepareIdealImage(const cv::Mat& image, std::array<cv::Point2f, 4> corners) {
	// TODO: This constant is from boardFinder internal. Make public so we can use the same on.
	static constexpr float WARP_SIZE = 1000.f; // Size of the warped board.

	// Desination rectangle
	std::array<cv::Point2f, 4> dst = {cv::Point2f(0, 0), cv::Point2f(WARP_SIZE, 0), cv::Point2f(WARP_SIZE, WARP_SIZE), cv::Point2f(0, WARP_SIZE)};

	cv::Mat H = cv::getPerspectiveTransform(corners.data(), dst.data());

	cv::Mat warped;
	cv::warpPerspective(image, warped, H, cv::Size(static_cast<int>(WARP_SIZE), static_cast<int>(WARP_SIZE)));
	return {warped, H, corners};
}

//! Expected grid spacing: ground-truth grid corners warped by \p H (which straightens the grid to an axis-aligned square), spread over boardSize-1 cells.
static double groundTruthSpacing(const GeometryGroundTruth& geometry, const cv::Mat& H) {
	// perspectiveTransform() needs a resizable point container; a fixed-size std::array as input trips an
	// OpenCV 5.0 assertion (NAryMatIterator size check) when it allocates the output.
	const std::vector<cv::Point2f> gridCorners(geometry.gridCorners.begin(), geometry.gridCorners.end());
	std::vector<cv::Point2f> warpedCorners;
	cv::perspectiveTransform(gridCorners, warpedCorners, H);

	const std::array<cv::Point2f, 4> corners = {warpedCorners[0], warpedCorners[1], warpedCorners[2], warpedCorners[3]};
	return boundingBoxMinDim(corners) / (geometry.boardSize - 1);
}

//! Check the geometry found by analyseGeometry(): same board size as the ground truth, and a spacing consistent with
//! the ground truth grid corners warped through the found homography.
void verifyBoardGeometry(const BoardGeometry& result, const GeometryGroundTruth& geometry) {
	EXPECT_EQ(result.boardSize, geometry.boardSize);

	const double spacing = groundTruthSpacing(geometry, result.H);
	EXPECT_NEAR(result.spacing, spacing, TOLERANCE_FRACTION * spacing);
}

//! Same check as verifyBoardGeometry(), against the geometry carried by the rectified board.
void verifyRectifiedBoard(const RectifiedBoard& board, const GeometryGroundTruth& geometry) {
	verifyBoardGeometry(board.geometry, geometry);
}


//! 1) Given an input image, prepares the image based on the defined board corner coordinates, then tests the gridFinder step with this prepared image.
void runIdealTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Load images file
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	// Get the corner points of the board
	for (const auto& imagePath: images) {
		// A) Prepare the test
		// Load image
		cv::Mat image = cv::imread(imagePath.string());
		ASSERT_FALSE(image.empty());

		// Load geometry information
		GeometryGroundTruth geometry = loadGeometryGroundTruth(imagePath);
		const float boardTolerance   = TOLERANCE_FRACTION * boundingBoxMinDim(geometry.boardCorners);

		// Transform with known corners
		const WarpResult warpResult = prepareIdealImage(image, geometry.boardCorners);
		ASSERT_TRUE(pointSetsMatch(warpResult.contourCorners, geometry.boardCorners, boardTolerance));

		// B) Start of the geometry test
		const BoardGeometry result = analyseGeometry(warpResult);
		EXPECT_TRUE(isValidGeometry(result));
		verifyBoardGeometry(result, geometry);

		// TODO: Can we do some tests on the transformed image?
		const RectifiedBoard board = transformImage(image, result);
		EXPECT_TRUE(isValidRectifiedBoard(board));
		verifyRectifiedBoard(board, geometry);
	}
}

//! 2) Given in input image, performs the boardFinder step, then tests the gridFinder step with the output of the boardFinder.
void runFullTest(std::string testSetName, unsigned imageCount) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	// Load images file
	const auto images = getImagesInDirectory(TEST_PATH);
	ASSERT_EQ(images.size(), imageCount);
	ensureJsonExists(images);

	// Get the corner points of the board
	for (const auto& imagePath: images) {
		// A) Prepare the test
		// Load image
		cv::Mat image = cv::imread(imagePath.string());
		ASSERT_FALSE(image.empty());

		// Load geometry information
		GeometryGroundTruth geometry = loadGeometryGroundTruth(imagePath);
		const float boardTolerance   = TOLERANCE_FRACTION * boundingBoxMinDim(geometry.boardCorners);

		// Transform with known corners
		const WarpResult warpResult = warpToBoard(image);
		ASSERT_TRUE(pointSetsMatch(warpResult.contourCorners, geometry.boardCorners, boardTolerance));

		// B) Start of the geometry test
		const BoardGeometry result = analyseGeometry(warpResult);
		EXPECT_TRUE(isValidGeometry(result));
		verifyBoardGeometry(result, geometry);

		// TODO: Can we do some tests on the transformed image?
		const RectifiedBoard board = transformImage(image, result);
		EXPECT_TRUE(isValidRectifiedBoard(board));
		verifyRectifiedBoard(board, geometry);
	}
}


// Ideal tests
TEST(GridFinder, Ideal_Angled_Easy) {
	runIdealTest("angled_easy", 6u);
}
TEST(GridFinder, Ideal_Angled_Hard) {
	runIdealTest("angled_hard", 6u);
}


// Full tests
TEST(GridFinder, Full_Angled_Easy) {
	runFullTest("angled_easy", 6u);
}
TEST(GridFinder, Full_Angled_Hard) {
	runFullTest("angled_hard", 6u);
}

} // namespace gtest
} // namespace tengen::vision::core
