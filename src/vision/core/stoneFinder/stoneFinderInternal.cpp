#include "stoneFinderInternal.hpp"

#include "decisionPolicy.hpp"
#include "refinementEngine.hpp"
#include "scoring.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace tengen::vision::core {

int roundedSpacingValue(double spacing, double scale, int fallback, int minValue, int maxValue) {
	int value = fallback;
	if (std::isfinite(spacing) && spacing > 0.0) {
		value = static_cast<int>(std::lround(spacing * scale));
	}
	return std::clamp(value, minValue, maxValue);
}

float computeNeighborMedianDelta(const std::vector<Features>& features, int gridX, int gridY, int boardSize, float fallback) {
	std::array<float, 8> neighborValues{};
	int count = 0;
	for (int deltaX = -1; deltaX <= 1; ++deltaX) {
		for (int deltaY = -1; deltaY <= 1; ++deltaY) {
			if (deltaX == 0 && deltaY == 0) {
				continue;
			}
			const int neighborX = gridX + deltaX;
			const int neighborY = gridY + deltaY;
			if (neighborX < 0 || neighborX >= boardSize || neighborY < 0 || neighborY >= boardSize) {
				continue;
			}

			const std::size_t neighborIndex = static_cast<std::size_t>(neighborX) * static_cast<std::size_t>(boardSize) + static_cast<std::size_t>(neighborY);
			if (neighborIndex >= features.size() || !features[neighborIndex].valid) {
				continue;
			}

			neighborValues[static_cast<std::size_t>(count++)] = features[neighborIndex].deltaL;
		}
	}

	if (count == 0) {
		return fallback;
	}
	std::sort(neighborValues.begin(), neighborValues.begin() + count);
	return (count % 2 == 1) ? neighborValues[static_cast<std::size_t>(count / 2)]
	                        : 0.5f * (neighborValues[static_cast<std::size_t>(count / 2 - 1)] + neighborValues[static_cast<std::size_t>(count / 2)]);
}

std::vector<float> computeNeighborMedianMap(const std::vector<Features>& features, int boardSize, float fallback) {
	std::vector<float> neighborMedianMap(features.size(), fallback);
	if (boardSize <= 0) {
		return neighborMedianMap;
	}
	for (std::size_t index = 0; index < features.size(); ++index) {
		const int gridX          = static_cast<int>(index) / boardSize;
		const int gridY          = static_cast<int>(index) - gridX * boardSize;
		neighborMedianMap[index] = computeNeighborMedianDelta(features, gridX, gridY, boardSize, fallback);
	}
	return neighborMedianMap;
}

int computeRefinementExtent(double spacing, const RefinementConfig& config) {
	return roundedSpacingValue(spacing, config.refineExtentSpacingK, config.refineExtentFallback, config.refineExtentMin, config.refineExtentMax);
}

void classifyAll(const std::vector<cv::Point2f>& intersections, const std::vector<Features>& features, const Model& model, unsigned boardSize,
                 const ScoringConfig& scoringConfig, const DecisionConfig& decisionConfig, const RefinementConfig& refinementConfig,
                 const RefinementEngine& refinementEngine, std::vector<StoneState>& outStates, std::vector<float>& outConfidence, DebugStats& outStats,
                 std::vector<Eval>* outEvaluations, std::vector<float>* outNeighborMedianMap, std::vector<RejectionReason>* outRejectionReasons) {
	outStates.assign(intersections.size(), StoneState::Empty);
	outConfidence.assign(intersections.size(), 0.0f);
	outStats = DebugStats{};
	(void)refinementConfig;
	if (outEvaluations != nullptr) {
		outEvaluations->assign(intersections.size(), Eval{});
	}
	if (outRejectionReasons != nullptr) {
		outRejectionReasons->assign(intersections.size(), RejectionReason::None);
	}

	const int boardSizeInt                     = static_cast<int>(boardSize);
	const std::vector<float> neighborMedianMap = computeNeighborMedianMap(features, boardSizeInt, model.medianEmpty);
	if (outNeighborMedianMap != nullptr) {
		*outNeighborMedianMap = neighborMedianMap;
	}
	const DecisionPolicy policy(decisionConfig);

	for (std::size_t index = 0; index < intersections.size(); ++index) {
		if (!features[index].valid) {
			if (outRejectionReasons != nullptr) {
				(*outRejectionReasons)[index] = RejectionReason::Other;
			}
			++outStats.emptyCount;
			continue;
		}

		Features feature = features[index];
		const SpatialContext context{Scoring::edgeLevel(index, boardSizeInt), neighborMedianMap[index], boardSize};

		const Eval baseEval = Scoring::evaluate(feature, model, context.edgeLevel, scoringConfig);
		RejectionReason rejectionReason{RejectionReason::None};
		const Eval baseDecision = policy.decide(feature, context, baseEval, outRejectionReasons != nullptr ? &rejectionReason : nullptr);
		Eval decision           = baseDecision;

		const DecisionPolicy::RefinementPath path = policy.refinementPath(feature, context, baseEval);
		if (path != DecisionPolicy::RefinementPath::None) {
			++outStats.refinedTried;
			if (policy.shouldRunRefinement(path, baseEval)) {
				Features refinedFeature = feature;
				Eval bestRawEval{};
				const bool refined = refinementEngine.searchBest(intersections[index], model, context, feature, baseEval, refinedFeature, bestRawEval);
				if (refined && policy.acceptsRefinement(path, baseEval, refinedFeature, bestRawEval)) {
					feature  = refinedFeature;
					decision = policy.decide(feature, context, bestRawEval, outRejectionReasons != nullptr ? &rejectionReason : nullptr);
					++outStats.refinedAccepted;
				}
			}
		}

		outStates[index]     = decision.state;
		outConfidence[index] = decision.confidence;
		if (outEvaluations != nullptr) {
			(*outEvaluations)[index] = baseEval;
		}
		if (outRejectionReasons != nullptr) {
			(*outRejectionReasons)[index] = (decision.state == StoneState::Empty) ? rejectionReason : RejectionReason::None;
		}
		if (decision.state == StoneState::Black) {
			++outStats.blackCount;
		} else if (decision.state == StoneState::White) {
			++outStats.whiteCount;
		} else {
			++outStats.emptyCount;
		}
	}
}

} // namespace tengen::vision::core
