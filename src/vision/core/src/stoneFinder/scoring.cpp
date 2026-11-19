#include "scoring.hpp"

#include <algorithm>
#include <cmath>

namespace tengen::vision::core::Scoring {

int edgeLevel(std::size_t index, int boardSize) {
	if (boardSize <= 0) {
		return 2;
	}
	const int gridX   = static_cast<int>(index) / boardSize;
	const int gridY   = static_cast<int>(index) - gridX * boardSize;
	const bool onEdge = (gridX == 0) || (gridX == boardSize - 1) || (gridY == 0) || (gridY == boardSize - 1);
	if (onEdge) {
		return 2;
	}
	const bool nearEdge = (gridX <= 1) || (gridX >= boardSize - 2) || (gridY <= 1) || (gridY >= boardSize - 2);
	return nearEdge ? 1 : 0;
}

Scores computeScores(const Features& feature, const Model& model, const ScoringConfig& config) {
	Scores scores{};
	scores.z                 = (feature.deltaL - model.medianEmpty) / model.sigmaEmpty;
	const float supportBlack = feature.darkFrac - feature.brightFrac;
	const float supportWhite = feature.brightFrac - feature.darkFrac;
	scores.chromaPenalty     = feature.chromaSq / (model.tChromaSq + feature.chromaSq);

	scores.black = config.scoreWDelta * (-scores.z) + config.scoreWSupport * supportBlack - config.scoreWChroma * scores.chromaPenalty;
	scores.white = config.scoreWDelta * scores.z + config.scoreWSupport * supportWhite - config.scoreWChroma * scores.chromaPenalty;
	scores.empty =
	        config.emptyScoreBias - config.emptyScoreZPenalty * std::abs(scores.z) - config.emptyScoreSupportPenalty * (feature.darkFrac + feature.brightFrac);
	return scores;
}

Eval evaluate(const Features& feature, const Model& model, int edgeLevelValue, const ScoringConfig& config) {
	const Scores scores = computeScores(feature, model, config);
	Eval result{};
	result.z = scores.z;

	StoneState bestState = StoneState::Black;
	float bestScore      = scores.black;
	float secondScore    = scores.white;

	if (scores.white > bestScore) {
		bestState   = StoneState::White;
		bestScore   = scores.white;
		secondScore = scores.black;
	}

	if (scores.empty > bestScore) {
		bestState   = StoneState::Empty;
		bestScore   = scores.empty;
		secondScore = std::max(scores.black, scores.white);
	} else if (scores.empty > secondScore) {
		secondScore = scores.empty;
	}

	result.state       = bestState;
	result.bestScore   = bestScore;
	result.secondScore = secondScore;
	result.margin      = result.bestScore - result.secondScore;
	result.required    = config.margin0 * (1.0f + config.edgePenalty * static_cast<float>(edgeLevelValue));
	result.confidence  = std::clamp(result.margin / (result.required + 1e-6f), 0.0f, 1.0f);
	result.confidence *= std::clamp(1.0f - config.confChromaDownweight * scores.chromaPenalty, 0.0f, 1.0f);
	result.confidence = std::clamp(result.confidence, 0.0f, 1.0f);
	return result;
}

} // namespace tengen::vision::core::Scoring
