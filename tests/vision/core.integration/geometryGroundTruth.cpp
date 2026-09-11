#include "geometryGroundTruth.hpp"

#include <cassert>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tengen::vision::core {
namespace gtest {

static std::array<cv::Point2f, 4> parsePoints(const nlohmann::json& array) {
	assert(array.size() == 4); // Malformed json data. Invalid test.

	std::size_t id = 0u;
	std::array<cv::Point2f, 4> points{};
	for (const auto& p: array) {
		points[id] = {p.at(0).get<float>(), p.at(1).get<float>()};
		++id;
	}
	return points;
}

//! Permutation of \p actual minimizing the total matching distance to \p expected.
static std::array<std::size_t, 4> bestMatchPermutation(const std::array<cv::Point2f, 4>& expected, const std::array<cv::Point2f, 4>& actual) {
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

GeometryGroundTruth GeometryGroundTruth::loadFromFile(const std::filesystem::path& jsonPath) {
	GeometryGroundTruth geometry{};

	std::ifstream file(jsonPath);
	if (!file.is_open()) {

		std::cerr << "Failed to open json file: " << jsonPath.string();
		return {};
	}

	const nlohmann::json j = nlohmann::json::parse(file, nullptr, false);
	EXPECT_FALSE(j.is_discarded()) << "Invalid JSON: " << jsonPath.string();

	geometry.boardSize    = j.value("boardSize", 0u);
	geometry.boardCorners = parsePoints(j.at("boardCorners"));
	geometry.gridCorners  = parsePoints(j.at("gridCorners"));
	return geometry;
}

bool pointSetsMatch(const std::array<cv::Point2f, 4>& expected, const std::array<cv::Point2f, 4>& actual, float tolerance) {
	const std::array<std::size_t, 4> bestPerm = bestMatchPermutation(expected, actual);

	bool match = true;
	for (std::size_t i = 0; i < expected.size(); ++i) {
		const double distance = cv::norm(expected[i] - actual[bestPerm[i]]);
		match &= distance <= tolerance;
	}
	return match;
}


//! Get the grid line spacing of the specified board geometry.
static double groundTruthSpacing(const GeometryGroundTruth& geometry, const cv::Mat& H) {
	// perspectiveTransform() needs a resizable point container; a fixed-size std::array as input trips an
	// OpenCV 5.0 assertion (NAryMatIterator size check) when it allocates the output.
	const std::vector<cv::Point2f> gridCorners(geometry.gridCorners.begin(), geometry.gridCorners.end());
	std::vector<cv::Point2f> warpedCorners;
	cv::perspectiveTransform(gridCorners, warpedCorners, H);

	const std::array<cv::Point2f, 4> corners = {warpedCorners[0], warpedCorners[1], warpedCorners[2], warpedCorners[3]};
	return minimumCornerPointDistance(corners) / (geometry.boardSize - 1);
}

void verifyBoardGeometry(const BoardGeometry& result, const GeometryGroundTruth& geometry, const float pointDeviationPercentage) {
	EXPECT_EQ(result.boardSize, geometry.boardSize);

	const double spacing = groundTruthSpacing(geometry, result.H);
	EXPECT_NEAR(result.spacing, spacing, pointDeviationPercentage * spacing);
}

void verifyRectifiedBoard(const RectifiedBoard& board, const GeometryGroundTruth& geometry) {
	// TODO: Can we test more maybe based on the image?
	verifyBoardGeometry(board.geometry, geometry);
}

//! Min-dimension of the axis-aligned bounding box of 4 points, in image-space pixels.
float minimumCornerPointDistance(const std::array<cv::Point2f, 4>& points) {
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


} // namespace gtest
} // namespace tengen::vision::core
