#include "testDataHelpers.hpp"

#include "core/serializer.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <numeric>

namespace tengen::vision::core {
namespace gtest {

static StoneState toStoneState(const Board::Stone stone) {
	switch (stone) {
	case Board::Stone::Black:
		return StoneState::Black;
	case Board::Stone::White:
		return StoneState::White;
	case Board::Stone::Empty:
		return StoneState::Empty;
	}
	return StoneState::Empty;
}

TestResult runPipeline(const std::filesystem::path& imgPath) {
	std::cout << "Running test: " << imgPath.string() << '\n';

	cv::Mat image = cv::imread(imgPath.string());
	EXPECT_FALSE(image.empty());

	// Warp image roughly around the board.
	WarpResult warped = warpToBoard(image);
	EXPECT_FALSE(warped.imageB0.empty());
	EXPECT_FALSE(warped.H0.empty());

	// Properly construct the board geometry.
	const BoardGeometry geometry = analyseGeometry(warped);
	EXPECT_TRUE(isValidGeometry(geometry));
	RectifiedBoard rectified = transformImage(image, geometry);
	EXPECT_TRUE(isValidRectifiedBoard(rectified));

	// Find the stones on the board.
	StoneResult stoneRes = analyseBoard(rectified);
	EXPECT_TRUE(stoneRes.success);
	EXPECT_EQ(stoneRes.stones.size(), rectified.geometry.intersections.size());

	return {warped, rectified, stoneRes};
}


Board loadExpectedBoard(const std::filesystem::path& imagePath) {
	Board expected(0u);
	std::filesystem::path txtPath = imagePath;
	txtPath.replace_extension(".txt");
	EXPECT_TRUE(readBoard(txtPath, expected)) << txtPath.string();
	return expected;
}

void expectStonesMatchBoard(const std::vector<StoneState>& stones, unsigned boardSize, const Board& expected) {
	ASSERT_EQ(stones.size(), static_cast<std::size_t>(boardSize) * boardSize);
	ASSERT_EQ(expected.size(), boardSize);

	for (unsigned x = 0; x < boardSize; ++x) {
		for (unsigned y = 0; y < boardSize; ++y) {
			const std::size_t index = static_cast<std::size_t>(x) * boardSize + y;
			EXPECT_EQ(stones[index], toStoneState(expected.get({x, y}))) << "Mismatch at (" << x << ", " << y << ")";
		}
	}
}


//! Permutation of \p actual minimizing the total matching distance to \p expected.
static std::array<std::size_t, 4> bestMatchPermutation(const std::vector<cv::Point2f>& expected, const std::vector<cv::Point2f>& actual) {
	std::array<std::size_t, 4> perm;
	std::iota(perm.begin(), perm.end(), std::size_t{0});

	std::array<std::size_t, 4> bestPerm = perm;
	double bestCost                     = std::numeric_limits<double>::max();
	do {
		double cost = 0.0;
		for (std::size_t i = 0; i < expected.size(); ++i) {
			cost += cv::norm(expected[i] - actual[perm[i]]);
		}
		if (cost < bestCost) {
			bestCost = cost;
			bestPerm = perm;
		}
	} while (std::next_permutation(perm.begin(), perm.end()));

	return bestPerm;
}

void expectPointsMatch(const std::vector<cv::Point2f>& expected, const std::vector<cv::Point2f>& actual, float tolerance, std::string_view context) {
	ASSERT_EQ(expected.size(), actual.size()) << context;

	const auto bestPerm = bestMatchPermutation(expected, actual);
	for (std::size_t i = 0; i < expected.size(); ++i) {
		const double distance = cv::norm(expected[i] - actual[bestPerm[i]]);
		EXPECT_LE(distance, tolerance) << context << ": point " << i << " off by " << distance << "px (tolerance " << tolerance << "px)";
	}
}

double maxMatchedPointDistance(const std::vector<cv::Point2f>& expected, const std::vector<cv::Point2f>& actual) {
	if (expected.empty() || expected.size() != actual.size()) {
		return std::numeric_limits<double>::infinity();
	}

	const auto bestPerm = bestMatchPermutation(expected, actual);

	double worst = 0.0;
	for (std::size_t i = 0; i < expected.size(); ++i) {
		worst = std::max(worst, cv::norm(expected[i] - actual[bestPerm[i]]));
	}
	return worst;
}

double quadIoU(const std::vector<cv::Point2f>& lhs, const std::array<cv::Point2f,4>& rhs) {
	if (lhs.size() < 3u || rhs.size() < 3u) {
		return 0.0;
	}

	// Hull both sides: cv::intersectConvexConvex() needs convex, consistently wound input, and the
	// ground truth corners are stored in no particular order.
	std::vector<cv::Point2f> hullLhs;
	std::vector<cv::Point2f> hullRhs;
	cv::convexHull(lhs, hullLhs);
	cv::convexHull(rhs, hullRhs);

	const double areaLhs = std::abs(cv::contourArea(hullLhs));
	const double areaRhs = std::abs(cv::contourArea(hullRhs));
	if (areaLhs <= 0.0 || areaRhs <= 0.0) {
		return 0.0;
	}

	std::vector<cv::Point2f> intersection;
	const double areaIntersection = cv::intersectConvexConvex(hullLhs, hullRhs, intersection, true);
	const double areaUnion        = areaLhs + areaRhs - areaIntersection;

	return areaUnion > 0.0 ? std::clamp(areaIntersection / areaUnion, 0.0, 1.0) : 0.0;
}

} // namespace gtest
} // namespace tengen::vision::core
