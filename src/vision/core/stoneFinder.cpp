#include "vision/core/stoneFinder.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <format>
#include <numeric>
#include <span>
#include <stdexcept>

namespace tengen::vision::core {

// Model input and output, see models/stoneNet.json.
static constexpr int PATCH_SIZE = 32; //!< Patch side, also the intersection spacing the patches are cut at.
static constexpr int OVERSAMPLE = 4;  //!< Warp at 4x, then shrink, so thin grid lines don't alias away.
//! In the order of the model's logits.
static constexpr std::array CLASSES{StoneState::Empty, StoneState::Black, StoneState::White};

//! imageB warped so intersections are PATCH_SIZE apart: intersection i then owns the cell (column, row) = (i / boardSize, i % boardSize).
static cv::Mat warpToPatchGrid(const RectifiedBoard& board) {
	const auto& points  = board.geometry.intersections;
	const std::size_t n = board.geometry.boardSize;
	const std::array from{points[0], points[(n - 1) * n], points[n * n - 1], points[n - 1]};

	const float first = OVERSAMPLE * PATCH_SIZE / 2.0f;
	const float last  = first + static_cast<float>(OVERSAMPLE * PATCH_SIZE) * static_cast<float>(n - 1);
	const std::array to{cv::Point2f{first, first}, cv::Point2f{last, first}, cv::Point2f{last, last}, cv::Point2f{first, last}};

	const int side = static_cast<int>(n) * PATCH_SIZE;
	cv::Mat oversampled;
	cv::warpPerspective(board.imageB, oversampled, cv::getPerspectiveTransform(from.data(), to.data()), cv::Size{side, side} * OVERSAMPLE, cv::INTER_LINEAR,
	                    cv::BORDER_REPLICATE);

	cv::Mat grid;
	cv::resize(oversampled, grid, cv::Size{side, side}, 0.0, 0.0, cv::INTER_AREA);
	return grid;
}

//! The patches of one board column, in the order of BoardGeometry::intersections. They are views into grid.
static std::vector<cv::Mat> cutColumn(const cv::Mat& grid, const int column, const int boardSize) {
	std::vector<cv::Mat> patches;
	patches.reserve(static_cast<std::size_t>(boardSize));
	for (int row = 0; row < boardSize; ++row) {
		patches.push_back(grid(cv::Rect{column * PATCH_SIZE, row * PATCH_SIZE, PATCH_SIZE, PATCH_SIZE}));
	}
	return patches;
}

//! Softmax probability of the largest logit.
static float softmaxOfBest(const std::span<const float> logits, const float best) {
	return 1.0f / std::transform_reduce(logits.begin(), logits.end(), 0.0f, std::plus{}, [best](const float logit) { return std::exp(logit - best); });
}

cv::dnn::Net loadStoneModel(const std::filesystem::path& modelPath) {
	cv::dnn::Net model;
	try {
		model = cv::dnn::readNetFromONNX(modelPath.string());
	} catch (const cv::Exception& e) {
		throw std::runtime_error(std::format("Cannot load stone model {}: {}", modelPath.string(), e.what()));
	}
	if (model.empty()) {
		throw std::runtime_error(std::format("Cannot load stone model {}: empty network", modelPath.string()));
	}
	return model;
}

StoneResult analyseBoard(const RectifiedBoard& board, cv::dnn::Net& model, DebugVisualizer* debugger) {
	if (!isValidRectifiedBoard(board) || board.imageB.type() != CV_8UC3) {
		return {.success = false, .stones = {}, .confidence = {}};
	}

	const int boardSize = static_cast<int>(board.geometry.boardSize);
	const cv::Mat grid  = warpToPatchGrid(board);

	StoneResult result{.success = true, .stones = {}, .confidence = {}};
	result.stones.reserve(board.geometry.intersections.size());
	result.confidence.reserve(board.geometry.intersections.size());

	// One column per forward pass: the model keeps the activations of a whole batch, about 170 kB per patch.
	for (int column = 0; column < boardSize; ++column) {
		model.setInput(cv::dnn::blobFromImages(cutColumn(grid, column, boardSize), 1.0 / 255.0, {}, {}, /*swapRB=*/true));
		const cv::Mat logits = model.forward();
		if (logits.type() != CV_32FC1 || logits.rows != boardSize || logits.cols != std::ssize(CLASSES)) {
			throw std::runtime_error("Stone model output does not match stoneNet.json");
		}

		for (int row = 0; row < boardSize; ++row) {
			const std::span<const float> scores{logits.ptr<float>(row), CLASSES.size()};
			const auto best = std::ranges::max_element(scores);
			result.stones.push_back(CLASSES[static_cast<std::size_t>(best - scores.begin())]);
			result.confidence.push_back(softmaxOfBest(scores, *best));
		}
	}

	if (debugger) {
		debugger->beginStage("Stone Detection");
		debugger->add("Patch Grid", grid);
		debugger->endStage();
	}
	return result;
}

} // namespace tengen::vision::core
