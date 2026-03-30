#pragma once

#include "stoneFinderTypes.hpp"

namespace tengen::vision::core {

class RefinementEngine;

int roundedSpacingValue(double spacing, double scale, int fallback, int minValue, int maxValue);
float computeNeighborMedianDelta(const std::vector<Features>& features, int gridX, int gridY, int boardSize, float fallback);
std::vector<float> computeNeighborMedianMap(const std::vector<Features>& features, int boardSize, float fallback);
int computeRefinementExtent(double spacing, const RefinementConfig& config);

void classifyAll(const std::vector<cv::Point2f>& intersections, const std::vector<Features>& features, const Model& model, unsigned boardSize,
                 const ScoringConfig& scoringConfig, const DecisionConfig& decisionConfig, const RefinementEngine& refinementEngine,
                 std::vector<StoneState>& outStates, std::vector<float>& outConfidence, DebugStats& outStats, std::vector<Eval>* outEvaluations = nullptr,
                 std::vector<float>* outNeighborMedianMap = nullptr, std::vector<RejectionReason>* outRejectionReasons = nullptr);

} // namespace tengen::vision::core
