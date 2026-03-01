#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace tengen::vision::core {
namespace gtest {

//! Create a synthetic, perfectly rectified board geometry with evenly spaced intersections.
static BoardGeometry makeSyntheticBoard(unsigned N, double spacingPx, const cv::Scalar& woodBgr) {
	BoardGeometry g{};
	g.boardSize = N;
	g.spacing   = spacingPx;
	g.H         = cv::Mat::eye(3, 3, CV_64F);

	const int margin = static_cast<int>(std::lround(spacingPx)); //!< keep ROIs fully inside the image
	const int span   = static_cast<int>(std::lround(spacingPx * static_cast<double>(N - 1)));
	const int w      = 2 * margin + span;
	const int h      = 2 * margin + span;
	g.imageB         = cv::Mat(h, w, CV_8UC3, woodBgr);

	g.intersections.reserve(static_cast<std::size_t>(N) * static_cast<std::size_t>(N));
	for (unsigned gx = 0; gx < N; ++gx) {
		for (unsigned gy = 0; gy < N; ++gy) {
			const float x = static_cast<float>(margin + std::lround(static_cast<double>(gx) * spacingPx));
			const float y = static_cast<float>(margin + std::lround(static_cast<double>(gy) * spacingPx));
			g.intersections.emplace_back(x, y);
		}
	}

	return g;
}

//! Draw a filled stone at a given grid coordinate (gx,gy).
static void drawStone(BoardGeometry& g, unsigned gx, unsigned gy, StoneState s) {
	ASSERT_FALSE(g.imageB.empty());
	ASSERT_GT(g.boardSize, 0u);
	ASSERT_EQ(g.intersections.size(), g.boardSize * g.boardSize);

	const unsigned idx = gx * g.boardSize + gy;
	ASSERT_LT(idx, g.intersections.size());

	const int r          = static_cast<int>(std::lround(g.spacing * 0.40)); //!< < 0.5*spacing to avoid overlap
	const cv::Scalar col = (s == StoneState::Black) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);
	cv::circle(g.imageB, g.intersections[idx], r, col, cv::FILLED, cv::LINE_AA);
}

//! Count occurrences of a given state.
static std::size_t countState(const std::vector<StoneState>& stones, StoneState s) {
	return static_cast<std::size_t>(std::count(stones.begin(), stones.end(), s));
}

static bool isStone(const StoneState state) {
	return state == StoneState::Black || state == StoneState::White;
}

static std::optional<StoneState> detectSingleStoneInImage(const std::filesystem::path& imagePath) {
	const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
	if (image.empty()) {
		return std::nullopt;
	}

	WarpResult warped = warpToBoard(image);
	if (!isValidBoard(warped)) {
		warped = {image, cv::Mat::eye(3, 3, CV_64F)};
	}

	const BoardGeometry geometry = rectifyImage(image, warped);
	if (!isValidGeometry(geometry)) {
		return std::nullopt;
	}

	const StoneResult result = analyseBoard(geometry);
	if (!result.success || result.stones.size() != geometry.intersections.size()) {
		return std::nullopt;
	}

	auto stoneIt = std::find_if(result.stones.begin(), result.stones.end(), isStone);
	if (stoneIt == result.stones.end()) {
		return std::nullopt;
	}

	const auto stoneIt2 = std::find_if(std::next(stoneIt), result.stones.end(), isStone);
	if (stoneIt2 != result.stones.end()) {
		return std::nullopt;
	}

	return *stoneIt;
}

TEST(StoneFinderUnit, EmptyBoard_NoStones) {
	BoardGeometry g = makeSyntheticBoard(9u, 80.0, cv::Scalar(80, 140, 200));

	const StoneResult r = analyseBoard(g);
	ASSERT_TRUE(r.success);
	ASSERT_EQ(r.stones.size(), g.intersections.size());
	ASSERT_EQ(r.confidence.size(), r.stones.size());

	EXPECT_EQ(countState(r.stones, StoneState::Black), 0u);
	EXPECT_EQ(countState(r.stones, StoneState::White), 0u);
}

TEST(StoneFinderUnit, SingleBlackStone_Detected) {
	BoardGeometry g = makeSyntheticBoard(9u, 80.0, cv::Scalar(80, 140, 200));
	drawStone(g, 4u, 4u, StoneState::Black);

	const StoneResult r = analyseBoard(g);
	ASSERT_TRUE(r.success);

	EXPECT_EQ(countState(r.stones, StoneState::Black), 1u);
	EXPECT_EQ(countState(r.stones, StoneState::White), 0u);
	EXPECT_EQ(r.stones[4u * 9u + 4u], StoneState::Black);
}

TEST(StoneFinderUnit, SingleWhiteStone_Detected) {
	BoardGeometry g = makeSyntheticBoard(9u, 80.0, cv::Scalar(80, 140, 200));
	drawStone(g, 4u, 4u, StoneState::White);

	const StoneResult r = analyseBoard(g);
	ASSERT_TRUE(r.success);

	EXPECT_EQ(countState(r.stones, StoneState::Black), 0u);
	EXPECT_EQ(countState(r.stones, StoneState::White), 1u);
	EXPECT_EQ(r.stones[4u * 9u + 4u], StoneState::White);
}

TEST(StoneFinderUnit, EdgeWhiteStone_Detected) {
	BoardGeometry g = makeSyntheticBoard(9u, 80.0, cv::Scalar(80, 140, 200));
	drawStone(g, 0u, 4u, StoneState::White); // on grid edge

	const StoneResult r = analyseBoard(g);
	ASSERT_TRUE(r.success);

	EXPECT_EQ(countState(r.stones, StoneState::Black), 0u);
	EXPECT_EQ(countState(r.stones, StoneState::White), 1u);
	EXPECT_EQ(r.stones[0u * 9u + 4u], StoneState::White);
}

TEST(StoneFinderUnit, BlackStone_WithMildGlare_NotWhite) {
	BoardGeometry g = makeSyntheticBoard(9u, 80.0, cv::Scalar(80, 140, 200));
	drawStone(g, 4u, 4u, StoneState::Black);

	// Add a small bright highlight inside the black stone (simulates mild glare/reflection).
	const cv::Point2f c = g.intersections[4u * 9u + 4u];
	cv::circle(g.imageB, c + cv::Point2f(10.0f, -10.0f), 6, cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);

	const StoneResult r = analyseBoard(g);
	ASSERT_TRUE(r.success);

	EXPECT_EQ(r.stones[4u * 9u + 4u], StoneState::Black);
	EXPECT_EQ(countState(r.stones, StoneState::White), 0u);
}

// Checks the board images which are just of a single stone and used to do the setup in the perception algorithm.
// The test is in this project because we check if the gauge stone can be detected. More complex checks on the same images are in the perception.gtest
TEST(StoneFinderUnit, SingleStoneSetup) {
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
		const auto detected  = detectSingleStoneInImage(imagePath);

		ASSERT_TRUE(detected.has_value()) << imagePath.string();
		EXPECT_EQ(*detected, expectedState) << imagePath.string();
	}
}

} // namespace gtest
} // namespace tengen::vision::core
