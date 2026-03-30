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

StoneResult analyseBoard(const BoardGeometry& geometry, DebugVisualizer* debugger, const StoneDetectionConfig& config) {
	if (geometry.imageB.empty()) {
		std::cerr << "Stone detection failed: input image is empty\n";
		return {false, {}, {}};
	}
	if (geometry.boardSize == 0u || geometry.intersections.size() != geometry.boardSize * geometry.boardSize) {
		std::cerr << "Stone detection failed: invalid board geometry\n";
		return {false, {}, {}};
	}

	if (debugger) {
		debugger->beginStage("Stone Detection v2");
		debugger->add("Input", geometry.imageB);
	}

	const Radii radii     = GeometrySampling::chooseRadii(geometry.spacing, config.geometry);
	const Offsets offsets = GeometrySampling::precomputeOffsets(radii);

	LabBlur blurredLab{};
	if (!FeatureExtraction::prepareLabBlur(geometry.imageB, radii, config.geometry, blurredLab)) {
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
		debugger->add("Stone Overlay", Debugging::drawOverlay(geometry.imageB, geometry.intersections, states, radii.innerRadius));
		debugger->add("Stone Stats", Debugging::renderStatsTile(model, stats));
		debugger->endStage();
	}

	return {true, std::move(states), std::move(confidence)};
}

} // namespace tengen::vision::core
