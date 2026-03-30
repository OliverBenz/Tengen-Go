#pragma once

#include "stoneFinderInternal.hpp"

namespace tengen::vision::core {

class RefinementEngine {
public:
	RefinementEngine(const SampleContext& sampleContext, const Offsets& offsets, const Radii& radii, double spacing, const GeometryConfig& geometryConfig,
	                 const ScoringConfig& scoringConfig, const RefinementConfig& refinementConfig);

	bool searchBest(const cv::Point2f& intersection, const Model& model, const SpatialContext& context, const Features& baseFeature, const Eval& baseEval,
	                Features& outFeature, Eval& outEval) const;

private:
	static bool isBetterCandidate(const Eval& currentBest, const Eval& candidate, const RefinementConfig& refinementConfig);

	const SampleContext& sampleContext_;
	const Offsets& offsets_;
	const Radii& radii_;
	double spacing_{0.0};
	const GeometryConfig& geometryConfig_;
	const ScoringConfig& scoringConfig_;
	const RefinementConfig& refinementConfig_;
};

} // namespace tengen::vision::core
