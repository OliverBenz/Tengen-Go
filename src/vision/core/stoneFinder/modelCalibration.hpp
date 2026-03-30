#pragma once

#include "stoneFinderInternal.hpp"

namespace tengen::vision::core::ModelCalibration {

float medianSorted(const std::vector<float>& sortedValues);
bool robustMedianSigma(const std::vector<float>& values, const CalibrationConfig& config, float& outMedian, float& outSigma);
bool calibrateModel(const std::vector<Features>& features, unsigned boardSize, const CalibrationConfig& calibrationConfig, Model& outModel);

} // namespace tengen::vision::core::ModelCalibration
