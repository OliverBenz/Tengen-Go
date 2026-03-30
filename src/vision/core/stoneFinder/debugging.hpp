#pragma once

#include "stoneFinderInternal.hpp"

namespace tengen::vision::core::Debugging {

bool isRuntimeDebugEnabled();
cv::Mat drawOverlay(const cv::Mat& image, const std::vector<cv::Point2f>& intersections, const std::vector<StoneState>& states, int radius);
cv::Mat renderStatsTile(const Model& model, const DebugStats& stats);
void emitRuntimeDebug(const BoardGeometry& geometry, const std::vector<Features>& features, const Model& model, const std::vector<StoneState>& states,
                      const std::vector<float>& confidence, const std::vector<Eval>& evaluations, const std::vector<float>& neighborMedianMap,
                      const DebugStats& stats, const std::vector<RejectionReason>* rejectionReasons = nullptr);

} // namespace tengen::vision::core::Debugging
