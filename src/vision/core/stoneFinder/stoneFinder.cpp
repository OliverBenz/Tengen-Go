#include "vision/core/stoneFinder.hpp"

#include "debugging.hpp"
#include "featureExtraction.hpp"
#include "geometrySampling.hpp"
#include "modelCalibration.hpp"
#include "refinementEngine.hpp"
#include "stoneFinderInternal.hpp"

#include <iostream>
#include <utility>

namespace tengen::vision::core {

StoneResult analyseBoard(const RectifiedBoard& board, DebugVisualizer* debugger, const StoneDetectionConfig& config) {
	if (!isValidRectifiedBoard(board)) {
		std::cerr << "Stone detection failed: invalid rectified board\n";
		return {false, {}, {}};
	}

	const BoardGeometry& geometry = board.geometry;

	if (debugger) {
		debugger->beginStage("Stone Detection v2");
		debugger->add("Input", board.imageB);
	}

	const Radii radii     = GeometrySampling::chooseRadii(geometry.spacing, config.geometry);
	const Offsets offsets = GeometrySampling::precomputeOffsets(radii);

	LabBlur blurredLab{};
	if (!FeatureExtraction::prepareLabBlur(board.imageB, radii, config.geometry, blurredLab)) {
		if (debugger) {
			debugger->endStage();
		}
		std::cerr << "Stone detection failed: unsupported channel count\n";
		return {false, {}, {}};
	}
	const SampleContext sampleContext{blurredLab.L, blurredLab.A, blurredLab.B, blurredLab.L.rows, blurredLab.L.cols};

	const std::vector<Features> features = FeatureExtraction::computeFeatures(geometry.intersections, sampleContext, offsets, radii, config.geometry);
	const RefinementEngine refinementEngine(sampleContext, offsets, radii, geometry.spacing, config.geometry, config.scoring, config.refinement);

	Model model{};
	if (!ModelCalibration::calibrateModel(features, geometry.boardSize, config.calibration, model)) {
		if (debugger) {
			debugger->endStage();
		}
		std::cerr << "Stone detection failed: calibration failed\n";
		return {false, {}, {}};
	}

	// Refactor-only: Behavior preserved. No functional changes.
	std::vector<StoneState> states;
	std::vector<float> confidence;
	DebugStats stats{};
	std::vector<Eval> evaluations;
	std::vector<float> neighborMedianMap;
	std::vector<RejectionReason> rejectionReasons;
	const bool collectRuntimeDebug = Debugging::isRuntimeDebugEnabled();
	classifyAll(geometry.intersections, features, model, geometry.boardSize, config.scoring, config.decision, refinementEngine, states, confidence, stats,
	            collectRuntimeDebug ? &evaluations : nullptr, collectRuntimeDebug ? &neighborMedianMap : nullptr,
	            collectRuntimeDebug ? &rejectionReasons : nullptr);

#if defined(VISION_DEBUG_LOGGING) && defined(VISION_LOG_STONEFINDER)
	Debugging::emitRuntimeDebug(geometry, features, model, states, confidence, evaluations, neighborMedianMap, stats,
	                            collectRuntimeDebug ? &rejectionReasons : nullptr);
#endif

	if (debugger) {
		debugger->add("Stone Overlay", Debugging::drawOverlay(board.imageB, geometry.intersections, states, radii.innerRadius));
		debugger->add("Stone Stats", Debugging::renderStatsTile(model, stats));
		debugger->endStage();
	}

	return {true, std::move(states), std::move(confidence)};
}

} // namespace tengen::vision::core
