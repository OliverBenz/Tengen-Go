#include "vision/core/gridFinder.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

namespace tengen::vision::core {
namespace gtest {

//! Draw a synthetic board image in canonical board coordinates.
static cv::Mat makeCanonicalBoardImage(unsigned boardSize, int sidePx) {
	cv::Mat board(sidePx, sidePx, CV_8UC3, cv::Scalar(145, 180, 208));

	const int margin     = static_cast<int>(std::lround(0.08 * static_cast<double>(sidePx)));
	const int gridSpan   = sidePx - 2 * margin;
	const double spacing = static_cast<double>(gridSpan) / static_cast<double>(boardSize - 1u);

	cv::rectangle(board, cv::Rect(margin, margin, gridSpan, gridSpan), cv::Scalar(45, 60, 70), 5, cv::LINE_AA);

	for (unsigned i = 0u; i < boardSize; ++i) {
		const int p = margin + static_cast<int>(std::lround(spacing * static_cast<double>(i)));
		cv::line(board, cv::Point(p, margin), cv::Point(p, margin + gridSpan), cv::Scalar(35, 45, 50), 2, cv::LINE_AA);
		cv::line(board, cv::Point(margin, p), cv::Point(margin + gridSpan, p), cv::Scalar(35, 45, 50), 2, cv::LINE_AA);
	}

	return board;
}

//! Warp a canonical board image into a scene using a destination quadrilateral.
static cv::Mat warpBoardToScene(const cv::Mat& board, cv::Size sceneSize, const std::array<cv::Point2f, 4>& dstQuad) {
	cv::Mat scene(sceneSize, CV_8UC3, cv::Scalar(25, 25, 25));

	const std::array<cv::Point2f, 4> srcQuad = {
	        cv::Point2f(0.0f, 0.0f),
	        cv::Point2f(static_cast<float>(board.cols - 1), 0.0f),
	        cv::Point2f(static_cast<float>(board.cols - 1), static_cast<float>(board.rows - 1)),
	        cv::Point2f(0.0f, static_cast<float>(board.rows - 1)),
	};

	const cv::Mat H = cv::getPerspectiveTransform(srcQuad.data(), dstQuad.data());

	cv::Mat warpedBoard;
	cv::warpPerspective(board, warpedBoard, H, scene.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	cv::Mat boardMask(board.rows, board.cols, CV_8UC1, cv::Scalar(255));
	cv::Mat warpedMask;
	cv::warpPerspective(boardMask, warpedMask, H, scene.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar::all(0));

	warpedBoard.copyTo(scene, warpedMask);
	return scene;
}

//! Build a high-contrast synthetic image with a board that almost touches all image borders.
static cv::Mat makeFullFrameSyntheticScene() {
	const cv::Mat board                      = makeCanonicalBoardImage(13u, 900);
	const std::array<cv::Point2f, 4> dstQuad = {
	        cv::Point2f(8.0f, 12.0f),
	        cv::Point2f(1272.0f, 6.0f),
	        cv::Point2f(1278.0f, 948.0f),
	        cv::Point2f(14.0f, 944.0f),
	};
	return warpBoardToScene(board, cv::Size(1280, 960), dstQuad);
}

//! Build a synthetic image with a clearly visible board outline but no internal grid.
static cv::Mat makeOutlineOnlySyntheticScene() {
	cv::Mat scene(900, 1300, CV_8UC3, cv::Scalar(20, 20, 20));
	const std::array<cv::Point, 4> quad = {cv::Point(12, 20), cv::Point(1284, 12), cv::Point(1288, 884), cv::Point(20, 892)};
	const std::vector<cv::Point> polygon(quad.begin(), quad.end());

	cv::fillConvexPoly(scene, polygon, cv::Scalar(205, 205, 205), cv::LINE_AA);
	cv::polylines(scene, polygon, true, cv::Scalar(35, 35, 35), 10, cv::LINE_AA);
	return scene;
}

// Check that the board can be detected. Same board different angles.
void runTest(const std::string& testSetName) {
	const auto TEST_PATH = std::filesystem::path(PATH_TEST_IMG) / testSetName;

	static constexpr unsigned IMG_COUNT  = 6u;
	static constexpr unsigned BOARD_SIZE = 13u;

	for (unsigned i = 1u; i <= IMG_COUNT; ++i) {
		std::string fileName = std::format("angle_{}.jpeg", i);

		cv::Mat image = cv::imread(TEST_PATH / fileName);
		ASSERT_FALSE(image.empty());

		const auto warpResult = warpToBoard(image);
		EXPECT_FALSE(warpResult.imageB0.empty());
		EXPECT_FALSE(warpResult.H0.empty());

		const auto geometry = rectifyImage(image, warpResult);
		EXPECT_FALSE(geometry.imageB.empty());
		EXPECT_FALSE(geometry.H.empty());
		EXPECT_EQ(geometry.intersections.size(), BOARD_SIZE * BOARD_SIZE);
		EXPECT_EQ(geometry.boardSize, BOARD_SIZE);
	}
}

TEST(Process, Find_Board_Easy) {
	runTest("angled_easy");
}

// TODO: These do not work yet.
TEST(Process, Find_Board_Hard) {
	runTest("angled_hard");
}

TEST(Process, Find_Board_Synthetic_FullFramePerspective) {
	const cv::Mat image = makeFullFrameSyntheticScene();
	ASSERT_FALSE(image.empty());

	const auto warpResult = warpToBoard(image);
	EXPECT_TRUE(isValidBoard(warpResult));

	const auto geometry = rectifyImage(image, warpResult);
	EXPECT_FALSE(geometry.imageB.empty());
	EXPECT_FALSE(geometry.H.empty());
	EXPECT_EQ(geometry.boardSize, 13u);
}

TEST(Process, Find_Board_Synthetic_OutlineOnly) {
	const cv::Mat image = makeOutlineOnlySyntheticScene();
	ASSERT_FALSE(image.empty());

	const auto warpResult = warpToBoard(image);
	EXPECT_TRUE(isValidBoard(warpResult));
}

} // namespace gtest
} // namespace tengen::vision::core
