#pragma once

#include "vision/core/debugVisualizer.hpp"
#include "vision/core/gridFinder.hpp"

#include <filesystem>
#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

namespace tengen::vision::core {

//! Stone state at a single grid intersection.
enum class StoneState {
	Empty,
	Black,
	White
};

//! Result of the stone detection stage.
struct StoneResult {
	bool success;                   //!< True if detection ran successfully; false on invalid input.
	std::vector<StoneState> stones; //!< Stone states aligned to RectifiedBoard::geometry.intersections (size = boardSize * boardSize).
	std::vector<float> confidence;  //!< Per-intersection softmax probability of stones[i] (size = stones.size()).
};

//! Load the stone CNN (models/stoneNet.onnx). Load once, reuse for every frame.
//! \throws std::runtime_error if modelPath is no readable ONNX model.
cv::dnn::Net loadStoneModel(const std::filesystem::path& modelPath);

/*! Detect stones on a Go board image in B(Board) space (see README).
 * \param [in]     board    Rectified board image and matching geometry.
 * \param [in,out] model    Network from loadStoneModel. Not thread-safe: a forward pass mutates it.
 * \param [in,out] debugger Optional debug visualizer for overlays.
 * \return         StoneResult where `stones[i]`/`confidence[i]` map to `board.geometry.intersections[i]`.
 */
StoneResult analyseBoard(const RectifiedBoard& board, cv::dnn::Net& model, DebugVisualizer* debugger = nullptr);

} // namespace tengen::vision::core
