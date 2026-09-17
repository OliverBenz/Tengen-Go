#include "testDataHelpers.hpp"

#include "core/serializer.hpp"

#include <algorithm>
#include <format>
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace tengen::vision::core {
namespace gtest {

std::vector<std::filesystem::path> getImagesInDirectory(const std::filesystem::path& directory) {
	std::vector<std::filesystem::path> images{};

	for (const auto& entry: std::filesystem::directory_iterator(directory)) {
		const auto extension = entry.path().extension();
		if (entry.is_regular_file() && (extension == ".jpeg" || extension == ".png")) {
			images.push_back(entry.path());
		}
	}

	std::sort(images.begin(), images.end()); // The file system order is unspecified, so sort to keep test runs comparable.
	return images;
}

void ensureJsonExists(const std::vector<std::filesystem::path>& images) {
	for (const auto& image: images) {
		const auto jsonFilePath = std::filesystem::path(image).replace_extension(".json");
		ASSERT_TRUE(std::filesystem::exists(jsonFilePath)) << jsonFilePath.string();
	}
}


WarpResult prepareIdealImage(const cv::Mat& image, const std::array<cv::Point2f, 4>& corners) {
	// TODO: This constant mirrors boardFinder internal::WARP_OUT_SIZE. Make it public so we can use the same one.
	static constexpr int WARP_SIZE = 1000; //!< Edge length of the image produced by the board warp.

	const std::array<cv::Point2f, 4> destination = {
	        cv::Point2f(0.f, 0.f),
	        cv::Point2f(WARP_SIZE - 1.f, 0.f),
	        cv::Point2f(WARP_SIZE - 1.f, WARP_SIZE - 1.f),
	        cv::Point2f(0.f, WARP_SIZE - 1.f),
	};

	const cv::Mat H = cv::getPerspectiveTransform(corners.data(), destination.data());

	cv::Mat warped;
	cv::warpPerspective(image, warped, H, cv::Size(WARP_SIZE, WARP_SIZE));
	return {warped, H, corners};
}

PipelineResult runPipeline(const cv::Mat& image) {
	return runPipeline(image, warpToBoard(image));
}

PipelineResult runPipeline(const cv::Mat& image, const WarpResult& warped) {
	const BoardGeometry geometry = analyseGeometry(warped);
	if (!isValidGeometry(geometry)) {
		return {warped, {}, {}}; // Stop here, the remaining stages cannot work without a grid.
	}

	const RectifiedBoard rectified = transformImage(image, geometry);
	return {warped, rectified, analyseBoard(rectified)};
}

bool isValidPipelineResult(const PipelineResult& result) {
	return isValidBoard(result.warped) && isValidRectifiedBoard(result.rectified) && result.stoneStep.success &&
	       result.stoneStep.stones.size() == result.rectified.geometry.intersections.size();
}


Board loadExpectedBoard(const std::filesystem::path& imagePath) {
	// Image series showing one and the same board share a single layout file.
	auto layoutPath = std::filesystem::path(imagePath).replace_extension(".txt");
	if (!std::filesystem::exists(layoutPath)) {
		layoutPath = imagePath.parent_path() / "board.txt";
	}

	Board expected(0u);
	EXPECT_TRUE(readBoard(layoutPath, expected)) << layoutPath.string();
	return expected;
}

//! Map a ground truth stone to the state the vision pipeline reports.
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

static std::string_view toString(const StoneState state) {
	switch (state) {
	case StoneState::Black:
		return "Black";
	case StoneState::White:
		return "White";
	case StoneState::Empty:
		return "Empty";
	}
	return "Empty";
}

//! Rotate a board coordinate by a multiple of 90 degrees.
static Coord rotateCoord(const Coord c, const unsigned quarterTurns, const unsigned boardSize) {
	const unsigned last = boardSize - 1u;

	switch (quarterTurns) {
	case 1u:
		return {last - c.y, c.x};
	case 2u:
		return {last - c.x, last - c.y};
	case 3u:
		return {c.y, last - c.x};
	}
	return c;
}

//! Stone states of a ground truth board seen from one of the four sides, laid out like StoneResult::stones.
static std::vector<StoneState> toStoneStates(const Board& board, const unsigned quarterTurns) {
	const auto boardSize = static_cast<unsigned>(board.size());

	std::vector<StoneState> stones(static_cast<std::size_t>(boardSize) * boardSize);
	for (unsigned x = 0; x < boardSize; ++x) {
		for (unsigned y = 0; y < boardSize; ++y) {
			stones[static_cast<std::size_t>(x) * boardSize + y] = toStoneState(board.get(rotateCoord({x, y}, quarterTurns, boardSize)));
		}
	}
	return stones;
}

//! A ground truth layout in one board orientation, together with where the detection differs from it.
struct OrientedLayout {
	std::vector<StoneState> stones;       //!< Ground truth layout in this orientation.
	std::vector<std::size_t> differences; //!< Intersections where the detection differs from it.
};

//! Compare a detection against the ground truth board seen in the given orientation.
static OrientedLayout compareToOrientation(const std::vector<StoneState>& stones, const Board& expected, const unsigned quarterTurns) {
	std::vector<StoneState> layout = toStoneStates(expected, quarterTurns);

	std::vector<std::size_t> differences{};
	for (std::size_t i = 0; i < stones.size(); ++i) {
		if (stones[i] != layout[i]) {
			differences.push_back(i);
		}
	}
	return {std::move(layout), std::move(differences)};
}

//! Orientation of the ground truth board the detection comes closest to.
//! \note The pipeline cannot know from which side the board was photographed, so all four rotations are valid results.
static OrientedLayout findClosestOrientation(const std::vector<StoneState>& stones, const Board& expected) {
	OrientedLayout closest = compareToOrientation(stones, expected, 0u);
	for (unsigned quarterTurns = 1u; quarterTurns < 4u; ++quarterTurns) {
		OrientedLayout candidate = compareToOrientation(stones, expected, quarterTurns);
		if (candidate.differences.size() < closest.differences.size()) {
			closest = std::move(candidate);
		}
	}
	return closest;
}

void expectStonesMatchBoard(const std::vector<StoneState>& stones, const Board& expected, const std::string_view context) {
	const auto boardSize = static_cast<unsigned>(expected.size());
	ASSERT_EQ(stones.size(), static_cast<std::size_t>(boardSize) * boardSize) << context;

	const OrientedLayout closest = findClosestOrientation(stones, expected);
	if (closest.differences.empty()) {
		return;
	}

	std::string report;
	for (const std::size_t index: closest.differences) {
		report += std::format("\n  ({}, {}): detected {}, expected {}", index / boardSize, index % boardSize, toString(stones[index]),
		                      toString(closest.stones[index]));
	}
	ADD_FAILURE() << closest.differences.size() << " of " << stones.size() << " intersections do not match the defined board layout in " << context << report;
}

} // namespace gtest
} // namespace tengen::vision::core
