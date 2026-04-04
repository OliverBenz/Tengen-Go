#pragma once

#include "vision/core/debugVisualizer.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinderConfig.hpp"

#include <opencv2/core/mat.hpp>
#include <vector>

namespace tengen::vision::core {

//! Stone state at a single grid intersection.
enum class StoneState { Empty, Black, White };

//! Result of the stone detection stage.
struct StoneResult {
	bool success;                   //!< True if detection ran successfully; false on invalid input.
	std::vector<StoneState> stones; //!< Stone states aligned to RectifiedBoard::geometry.intersections (size = boardSize * boardSize).
	std::vector<float> confidence;  //!< Per-intersection confidence for stones[i] (size = stones.size()). 0 -> Empty/unknown.
};

/*! Detect stones on a Go board image in B(Board) space (see README).
 * \param [in]     board    Rectified board image and matching geometry.
 * \param [in,out] debugger Optional debug visualizer for overlays.
 * \param [in]     config   Stone detection configuration.
 * \return         StoneResult where `stones[i]`/`confidence[i]` map to `board.geometry.intersections[i]`.
 */
StoneResult analyseBoard(const RectifiedBoard& board, DebugVisualizer* debugger = nullptr, const StoneDetectionConfig& config = StoneDetectionConfig{});

} // namespace tengen::vision::core
