#pragma once

#include "vision/core/debugVisualizer.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneClassifier.hpp"

#include <vector>

namespace tengen::vision::core {

//! Result of the stone detection stage.
struct StoneResult {
	bool success;                   //!< True if detection ran successfully; false on invalid input.
	std::vector<StoneState> stones; //!< Stone states aligned to RectifiedBoard::geometry.intersections (size = boardSize * boardSize).
	std::vector<float> confidence;  //!< Per-intersection softmax probability of stones[i] (size = stones.size()).
};

/*! Detect stones on a Go board image in B(Board) space (see README).
 * \param [in]     board      Rectified board image and matching geometry.
 * \param [in,out] classifier Reuse for every board.
 * \param [in,out] debugger   Optional debug visualizer for overlays.
 * \return         StoneResult where `stones[i]`/`confidence[i]` map to `board.geometry.intersections[i]`.
 */
StoneResult analyseBoard(const RectifiedBoard& board, StoneClassifier& classifier, DebugVisualizer* debugger = nullptr);

} // namespace tengen::vision::core
