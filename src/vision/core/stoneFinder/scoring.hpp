#pragma once

#include "stoneFinderTypes.hpp"

namespace tengen::vision::core::Scoring {

int edgeLevel(std::size_t index, int boardSize);
Scores computeScores(const Features& feature, const Model& model, const ScoringConfig& config);
Eval evaluate(const Features& feature, const Model& model, int edgeLevelValue, const ScoringConfig& config);

} // namespace tengen::vision::core::Scoring
