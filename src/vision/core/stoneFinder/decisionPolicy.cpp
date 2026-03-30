#include "decisionPolicy.hpp"

#include <cmath>

namespace tengen::vision::core {

DecisionPolicy::DecisionPolicy(const DecisionConfig& decisionConfig) : decisionConfig_(decisionConfig) {
}

Eval DecisionPolicy::decide(const Features& feature, const SpatialContext& context, const Eval& evaluated, RejectionReason* outReason) const {
	if (outReason != nullptr) {
		*outReason = RejectionReason::None;
	}
	RejectionReason rejectionReason{RejectionReason::None};
	if (!passesStatistical(evaluated, context, &rejectionReason) || !passesSupport(evaluated, feature, context, &rejectionReason) ||
	    !passesEdge(evaluated, feature, context, &rejectionReason) || !passesMargin(evaluated, &rejectionReason)) {
		if (outReason != nullptr) {
			*outReason = rejectionReason;
		}
		return rejected(evaluated);
	}
	return evaluated;
}

DecisionPolicy::RefinementPath DecisionPolicy::refinementPath(const Features& feature, const SpatialContext&, const Eval& evaluated) const {
	if (hasEmptyRescueHint(feature, evaluated)) {
		return RefinementPath::EmptyRescue;
	}
	if (hasStandardRefineHint(feature, evaluated)) {
		return RefinementPath::Standard;
	}
	return RefinementPath::None;
}

bool DecisionPolicy::shouldRunRefinement(RefinementPath path, const Eval& evaluated) const {
	if (path == RefinementPath::None) {
		return false;
	}
	if (evaluated.state == StoneState::Empty) {
		return evaluated.margin < decisionConfig_.refineSkipStableEmptyMarginMult * evaluated.required;
	}
	return evaluated.margin < decisionConfig_.refineTriggerMult * evaluated.required;
}

bool DecisionPolicy::acceptsRefinement(RefinementPath path, const Eval& baseEval, const Features& refinedFeature, const Eval& refinedEval) const {
	if (path == RefinementPath::None) {
		return false;
	}
	if (path == RefinementPath::Standard) {
		return refinedEval.margin > baseEval.margin + decisionConfig_.refineAcceptGainMult * baseEval.required;
	}
	const float minGain = decisionConfig_.refineAcceptFromEmptyGainMult * baseEval.required;
	return refinedEval.state == StoneState::White && refinedEval.margin > baseEval.margin + minGain &&
	       refinedEval.margin >= decisionConfig_.emptyRescueMinMarginMult * refinedEval.required &&
	       (refinedFeature.brightFrac - refinedFeature.darkFrac) >= decisionConfig_.minSupportAdvantageWhite;
}

bool DecisionPolicy::passesStatistical(const Eval& evaluated, const SpatialContext& context, RejectionReason* outReason) const {
	if (isWeakByZ(evaluated.state, evaluated.z, context.edgeLevel)) {
		return fail(outReason, RejectionReason::WeakZ);
	}
	if (evaluated.state == StoneState::Black && failsBlackConfidence(evaluated, context.boardSize)) {
		return fail(outReason, RejectionReason::LowConfidence);
	}
	if (evaluated.state == StoneState::White && failsWhiteConfidence(evaluated, context.boardSize)) {
		return fail(outReason, RejectionReason::LowConfidence);
	}
	return true;
}

bool DecisionPolicy::passesSupport(const Eval& evaluated, const Features& feature, const SpatialContext& context, RejectionReason* outReason) const {
	if (evaluated.state == StoneState::Black) {
		const bool weakSupport  = feature.darkFrac < decisionConfig_.minSupportBlack;
		const bool weakContrast = (feature.darkFrac - feature.brightFrac) < decisionConfig_.minSupportAdvantageBlack;
		const bool weakNeighbor = (context.neighborMedian - feature.deltaL) < decisionConfig_.minNeighborContrastBlack;
		if (weakSupport || weakContrast) {
			return fail(outReason, RejectionReason::WeakSupport);
		}
		if (weakNeighbor) {
			return fail(outReason, RejectionReason::WeakNeighborContrast);
		}
	}
	if (evaluated.state == StoneState::White && failsWhiteSupport(feature, context, evaluated.z)) {
		return fail(outReason, RejectionReason::WeakSupport);
	}
	return true;
}

bool DecisionPolicy::passesEdge(const Eval& evaluated, const Features& feature, const SpatialContext& context, RejectionReason* outReason) const {
	if (evaluated.state == StoneState::White && failsWhiteEdgeSanity(feature, context, evaluated.confidence)) {
		return fail(outReason, RejectionReason::EdgeArtifact);
	}
	return true;
}

bool DecisionPolicy::passesMargin(const Eval& evaluated, RejectionReason* outReason) const {
	if (evaluated.state == StoneState::Black && evaluated.margin < decisionConfig_.minBlackMarginMult * evaluated.required) {
		return fail(outReason, RejectionReason::MarginTooSmall);
	}
	if (evaluated.state == StoneState::White && evaluated.margin < decisionConfig_.minWhiteMarginMult * evaluated.required) {
		return fail(outReason, RejectionReason::MarginTooSmall);
	}
	return true;
}

Eval DecisionPolicy::rejected(const Eval& decision) {
	Eval rejectedDecision       = decision;
	rejectedDecision.state      = StoneState::Empty;
	rejectedDecision.confidence = 0.0f;
	return rejectedDecision;
}

bool DecisionPolicy::fail(RejectionReason* outReason, RejectionReason reason) {
	if (outReason != nullptr) {
		*outReason = reason;
	}
	return false;
}

bool DecisionPolicy::isWeakByZ(StoneState state, float z, int edgeLevelValue) const {
	const float minBlackZ = decisionConfig_.minZBlack +
	                        (edgeLevelValue == 1 ? decisionConfig_.minZBlackNearEdgeAdd : (edgeLevelValue == 2 ? decisionConfig_.minZBlackOnEdgeAdd : 0.0f));
	return (state == StoneState::Black && (-z) < minBlackZ) || (state == StoneState::White && z < decisionConfig_.minZWhite);
}

bool DecisionPolicy::failsBlackConfidence(const Eval& decision, unsigned boardSize) const {
	const float threshold = (boardSize >= decisionConfig_.minConfidenceBlackBoardSize) ? decisionConfig_.minConfidenceBlack : 0.0f;
	return decision.confidence < threshold;
}

bool DecisionPolicy::failsWhiteConfidence(const Eval& decision, unsigned boardSize) const {
	const float threshold = (boardSize >= decisionConfig_.minConfidenceWhiteBoardSize) ? decisionConfig_.minConfidenceWhite : 0.0f;
	return decision.confidence < threshold;
}

bool DecisionPolicy::hasStrongWhiteSupport(const Features& feature, const SpatialContext& context) const {
	const float brightAdvantage  = feature.brightFrac - feature.darkFrac;
	const float neighborContrast = feature.deltaL - context.neighborMedian;
	return brightAdvantage >= decisionConfig_.whiteStrongAdvMin && neighborContrast >= decisionConfig_.whiteStrongNeighborMin;
}

bool DecisionPolicy::qualifiesLowChromaRescue(const Features& feature, const SpatialContext& context, float z) const {
	const float chromaCap = (context.edgeLevel == 1) ? decisionConfig_.whiteLowChromaMaxNearEdge : decisionConfig_.whiteLowChromaMax;
	const float minBright = (context.edgeLevel == 1) ? decisionConfig_.whiteLowChromaMinBrightNearEdge : decisionConfig_.whiteLowChromaMinBright;
	if (feature.chromaSq > chromaCap || z < decisionConfig_.whiteLowChromaMinZ || feature.brightFrac < minBright) {
		return false;
	}
	constexpr float LOW_CHROMA_BAND_MAX = 90.0f;
	const bool inLowBand                = feature.chromaSq <= LOW_CHROMA_BAND_MAX;
	const bool inHighBand               = feature.chromaSq >= (0.75f * chromaCap);
	if (!inLowBand && !inHighBand) {
		return false;
	}
	if (context.edgeLevel == 1) {
		return inHighBand;
	}
	return true;
}

bool DecisionPolicy::failsWhiteSupport(const Features& feature, const SpatialContext& context, float z) const {
	const float brightAdvantage = feature.brightFrac - feature.darkFrac;
	if (brightAdvantage < decisionConfig_.minSupportAdvantageWhiteFloor) {
		return true;
	}
	if (feature.brightFrac < decisionConfig_.minSupportWhite) {
		return true;
	}
	return !hasStrongWhiteSupport(feature, context) && !qualifiesLowChromaRescue(feature, context, z);
}

bool DecisionPolicy::isNearEdgeColorArtifact(const Features& feature, const SpatialContext& context) const {
	return context.edgeLevel == 1 && feature.chromaSq >= decisionConfig_.edgeWhiteNearChromaSq &&
	       feature.brightFrac < decisionConfig_.edgeWhiteNearMinBrightFrac;
}

bool DecisionPolicy::isOnEdgeColorArtifact(const Features& feature, const SpatialContext& context) const {
	return context.edgeLevel >= 2 && feature.chromaSq >= decisionConfig_.edgeWhiteHighChromaSq && feature.brightFrac < decisionConfig_.edgeWhiteMinBrightFrac;
}

bool DecisionPolicy::isEdgeColorArtifact(const Features& feature, const SpatialContext& context) const {
	return isNearEdgeColorArtifact(feature, context) || isOnEdgeColorArtifact(feature, context);
}

bool DecisionPolicy::isNearEdgeUnstableWhite(const Features& feature, const SpatialContext& context, float confidence) const {
	return context.edgeLevel == 1 && feature.chromaSq >= decisionConfig_.edgeWhiteNearWeakChromaSq &&
	       feature.brightFrac < decisionConfig_.edgeWhiteNearWeakBrightFrac && confidence < decisionConfig_.edgeWhiteNearWeakMinConf;
}

bool DecisionPolicy::failsWhiteEdgeSanity(const Features& feature, const SpatialContext& context, float confidence) const {
	return isEdgeColorArtifact(feature, context) || isNearEdgeUnstableWhite(feature, context, confidence);
}

bool DecisionPolicy::hasEmptyRescueHint(const Features& feature, const Eval& evaluated) const {
	if (evaluated.state != StoneState::Empty) {
		return false;
	}
	const float brightAdvantage = feature.brightFrac - feature.darkFrac;
	return evaluated.z >= decisionConfig_.emptyRescueMinZ && feature.brightFrac >= decisionConfig_.emptyRescueMinBright &&
	       brightAdvantage >= decisionConfig_.emptyRescueMinBrightAdv;
}

bool DecisionPolicy::hasStandardRefineHint(const Features& feature, const Eval& evaluated) const {
	if (evaluated.state == StoneState::Empty) {
		return false;
	}
	const float baseSupportAdv = (evaluated.state == StoneState::Black) ? (feature.darkFrac - feature.brightFrac) : (feature.brightFrac - feature.darkFrac);
	const float minAbsZ        = (evaluated.state == StoneState::White) ? decisionConfig_.refineMinAbsZWhite : decisionConfig_.refineMinAbsZBlack;
	const float minSupportAdv  = (evaluated.state == StoneState::White) ? decisionConfig_.refineMinSupportAdvWhite : decisionConfig_.refineMinSupportAdvBlack;
	const bool allowed         = std::abs(evaluated.z) >= minAbsZ && baseSupportAdv >= minSupportAdv;
	return allowed && evaluated.margin < decisionConfig_.refineTriggerMult * evaluated.required;
}

} // namespace tengen::vision::core
