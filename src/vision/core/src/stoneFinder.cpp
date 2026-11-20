#include "vision/core/stoneFinder.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cstddef>

namespace tengen::vision::core {

static constexpr int PATCH      = StoneClassifier::PATCH_SIZE;
static constexpr int OVERSAMPLE = 4; //!< Keeps thin grid lines from aliasing.

//! imageB warped to PATCH spacing: intersection i is cell (i / n, i % n).
static cv::Mat toPatchGrid(const RectifiedBoard& board) {
	const std::vector<cv::Point2f>& p = board.geometry.intersections;
	const std::size_t n               = board.geometry.boardSize;
	const std::array<cv::Point2f, 4> from{p[0], p[(n - 1) * n], p[n * n - 1], p[n - 1]}; // TL, TR, BR, BL.

	const float first = OVERSAMPLE * PATCH / 2.f;
	const float last  = first + static_cast<float>(OVERSAMPLE * PATCH) * static_cast<float>(n - 1);
	const std::array<cv::Point2f, 4> to{{{first, first}, {last, first}, {last, last}, {first, last}}};

	const int side = static_cast<int>(n) * PATCH;
	cv::Mat fine;
	cv::warpPerspective(board.imageB, fine, cv::getPerspectiveTransform(from.data(), to.data()), {side * OVERSAMPLE, side * OVERSAMPLE}, cv::INTER_LINEAR,
	                    cv::BORDER_REPLICATE);
	cv::Mat grid;
	cv::resize(fine, grid, {side, side}, 0.0, 0.0, cv::INTER_AREA);
	return grid;
}

//! Every stone found circled: red for black, blue for white.
static cv::Mat drawStones(const RectifiedBoard& board, const std::vector<StoneState>& stones) {
	cv::Mat marked     = board.imageB.clone();
	const int radius   = static_cast<int>(board.geometry.spacing * 0.45);
	const auto& points = board.geometry.intersections;
	for (std::size_t i = 0; i < stones.size(); ++i) {
		if (stones[i] != StoneState::Empty) {
			const cv::Scalar colour = stones[i] == StoneState::Black ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
			cv::circle(marked, points[i], radius, colour, 2, cv::LINE_AA);
		}
	}
	return marked;
}

StoneResult analyseBoard(const RectifiedBoard& board, StoneClassifier& classifier, DebugVisualizer* debugger) {
	if (!isValidRectifiedBoard(board) || board.imageB.type() != CV_8UC3) {
		return {false, {}, {}};
	}

	const cv::Mat grid = toPatchGrid(board);
	const int n        = static_cast<int>(board.geometry.boardSize);
	std::vector<cv::Mat> patches;
	patches.reserve(board.geometry.intersections.size());
	for (int column = 0; column < n; ++column) {
		for (int row = 0; row < n; ++row) {
			patches.push_back(grid(cv::Rect(column * PATCH, row * PATCH, PATCH, PATCH)));
		}
	}

	StoneResult result{true, {}, {}};
	for (const Classification& classification: classifier.classify(patches)) {
		result.stones.push_back(classification.state);
		result.confidence.push_back(classification.confidence);
	}

	if (debugger) {
		debugger->beginStage("Stone Detection");
		debugger->add("Patch Grid", grid);
		debugger->add("Stones", drawStones(board, result.stones));
		debugger->endStage();
	}
	return result;
}

} // namespace tengen::vision::core
