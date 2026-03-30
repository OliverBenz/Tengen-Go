#include "refinementEngine.hpp"

#include "featureExtraction.hpp"
#include "scoring.hpp"
#include "stoneFinderInternal.hpp"

#include <cmath>

namespace tengen::vision::core {

RefinementEngine::RefinementEngine(const SampleContext& sampleContext, const Offsets& offsets, const Radii& radii, double spacing,
                                   const GeometryConfig& geometryConfig, const ScoringConfig& scoringConfig, const RefinementConfig& refinementConfig)
    : sampleContext_(sampleContext), offsets_(offsets), radii_(radii), spacing_(spacing), geometryConfig_(geometryConfig), scoringConfig_(scoringConfig),
      refinementConfig_(refinementConfig) {
}

bool RefinementEngine::searchBest(const cv::Point2f& intersection, const Model& model, const SpatialContext& context, const Features& baseFeature,
                                  const Eval& baseEval, Features& outFeature, Eval& outEval) const {
	const int extent  = computeRefinementExtent(spacing_, refinementConfig_);
	const int centerX = static_cast<int>(std::lround(intersection.x));
	const int centerY = static_cast<int>(std::lround(intersection.y));

	bool sampledAny      = false;
	Eval bestEval        = baseEval;
	Features bestFeature = baseFeature;
	for (int offsetY = -extent; offsetY <= extent; offsetY += refinementConfig_.refineStepPx) {
		for (int offsetX = -extent; offsetX <= extent; offsetX += refinementConfig_.refineStepPx) {
			if (offsetX == 0 && offsetY == 0) {
				continue;
			}
			Features candidateFeature{};
			if (!FeatureExtraction::computeFeaturesAt(sampleContext_, offsets_, radii_, geometryConfig_, centerX + offsetX, centerY + offsetY,
			                                          candidateFeature)) {
				continue;
			}
			sampledAny               = true;
			const Eval candidateEval = Scoring::evaluate(candidateFeature, model, context.edgeLevel, scoringConfig_);
			if (isBetterCandidate(bestEval, candidateEval, refinementConfig_)) {
				bestEval    = candidateEval;
				bestFeature = candidateFeature;
			}
		}
	}
	outFeature = bestFeature;
	outEval    = bestEval;
	return sampledAny;
}

bool RefinementEngine::isBetterCandidate(const Eval& currentBest, const Eval& candidate, const RefinementConfig& refinementConfig) {
	const bool betterMargin      = candidate.margin > currentBest.margin;
	const bool promotesFromEmpty = (currentBest.state == StoneState::Empty) && (candidate.state != StoneState::Empty) &&
	                               (candidate.margin + refinementConfig.refinePromoteFromEmptyEps >= currentBest.margin);
	return betterMargin || promotesFromEmpty;
}

} // namespace tengen::vision::core
