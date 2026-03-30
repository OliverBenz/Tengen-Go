#pragma once

#include "stoneFinderTypes.hpp"

namespace tengen::vision::core {

class DecisionPolicy {
public:
	enum class RefinementPath { None, EmptyRescue, Standard };

	explicit DecisionPolicy(const DecisionConfig& decisionConfig);

	Eval decide(const Features& feature, const SpatialContext& context, const Eval& evaluated, RejectionReason* outReason = nullptr) const;
	RefinementPath refinementPath(const Features& feature, const SpatialContext& context, const Eval& evaluated) const;
	bool shouldRunRefinement(RefinementPath path, const Eval& evaluated) const;
	bool acceptsRefinement(RefinementPath path, const Eval& baseEval, const Features& refinedFeature, const Eval& refinedEval) const;

	bool passesStatistical(const Eval& evaluated, const SpatialContext& context, RejectionReason* outReason = nullptr) const;
	bool passesSupport(const Eval& evaluated, const Features& feature, const SpatialContext& context, RejectionReason* outReason = nullptr) const;
	bool passesEdge(const Eval& evaluated, const Features& feature, const SpatialContext& context, RejectionReason* outReason = nullptr) const;
	bool passesMargin(const Eval& evaluated, RejectionReason* outReason = nullptr) const;

private:
	const DecisionConfig& decisionConfig_;

	static Eval rejected(const Eval& decision);
	static bool fail(RejectionReason* outReason, RejectionReason reason);

	bool isWeakByZ(StoneState state, float z, int edgeLevelValue) const;
	bool failsBlackConfidence(const Eval& decision, unsigned boardSize) const;
	bool failsWhiteConfidence(const Eval& decision, unsigned boardSize) const;
	bool hasStrongWhiteSupport(const Features& feature, const SpatialContext& context) const;
	bool qualifiesLowChromaRescue(const Features& feature, const SpatialContext& context, float z) const;
	bool failsWhiteSupport(const Features& feature, const SpatialContext& context, float z) const;
	bool isNearEdgeColorArtifact(const Features& feature, const SpatialContext& context) const;
	bool isOnEdgeColorArtifact(const Features& feature, const SpatialContext& context) const;
	bool isEdgeColorArtifact(const Features& feature, const SpatialContext& context) const;
	bool isNearEdgeUnstableWhite(const Features& feature, const SpatialContext& context, float confidence) const;
	bool failsWhiteEdgeSanity(const Features& feature, const SpatialContext& context, float confidence) const;
	bool hasEmptyRescueHint(const Features& feature, const Eval& evaluated) const;
	bool hasStandardRefineHint(const Features& feature, const Eval& evaluated) const;
};

} // namespace tengen::vision::core
