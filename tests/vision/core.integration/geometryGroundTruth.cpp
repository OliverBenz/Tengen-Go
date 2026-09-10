#include "geometryGroundTruth.hpp"

#include <cassert>
#include <fstream>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tengen::vision::core {
namespace gtest {

static std::array<cv::Point2f, 4> parsePoints(const nlohmann::json& array) {
	assert(array.size()== 4); // Malformed json data. Invalid test.

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

GeometryGroundTruth loadGeometryGroundTruth(const std::filesystem::path& imagePath) {
	GeometryGroundTruth geometry{};

	std::filesystem::path jsonPath = imagePath;
	jsonPath.replace_extension(".json");

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

} // namespace gtest
} // namespace tengen::vision::core
