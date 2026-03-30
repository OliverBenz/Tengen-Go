#include "modelCalibration.hpp"

#include <algorithm>
#include <cmath>

namespace tengen::vision::core::ModelCalibration {

float medianSorted(const std::vector<float>& sortedValues) {
	const std::size_t count = sortedValues.size();
	if (count == 0u) {
		return 0.0f;
	}
	return (count % 2u == 1u) ? sortedValues[(count - 1u) / 2u] : 0.5f * (sortedValues[count / 2u - 1u] + sortedValues[count / 2u]);
}

bool robustMedianSigma(const std::vector<float>& values, const CalibrationConfig& config, float& outMedian, float& outSigma) {
	if (values.empty()) {
		return false;
	}

	std::vector<float> sortedValues = values;
	std::sort(sortedValues.begin(), sortedValues.end());
	outMedian = medianSorted(sortedValues);

	std::vector<float> absoluteDeviation;
	absoluteDeviation.reserve(sortedValues.size());
	for (float value: sortedValues) {
		absoluteDeviation.push_back(std::abs(value - outMedian));
	}
	std::sort(absoluteDeviation.begin(), absoluteDeviation.end());

	const float mad = medianSorted(absoluteDeviation);
	outSigma        = std::max(config.sigmaMin, config.madToSigma * mad);
	return true;
}

bool calibrateModel(const std::vector<Features>& features, unsigned boardSize, const CalibrationConfig& calibrationConfig, Model& outModel) {
	std::vector<float> allDelta;
	std::vector<float> allChroma;
	allDelta.reserve(features.size());
	allChroma.reserve(features.size());
	for (const Features& feature: features) {
		if (!feature.valid) {
			continue;
		}
		allDelta.push_back(feature.deltaL);
		allChroma.push_back(feature.chromaSq);
	}
	if (allDelta.empty()) {
		return false;
	}

	float medianInitial = 0.0f;
	float sigmaInitial  = calibrationConfig.sigmaMin;
	if (!robustMedianSigma(allDelta, calibrationConfig, medianInitial, sigmaInitial)) {
		return false;
	}

	const float emptyBand = calibrationConfig.emptyBandSigma * sigmaInitial;
	std::vector<float> likelyEmptyDelta;
	std::vector<float> likelyEmptyChroma;
	likelyEmptyDelta.reserve(allDelta.size());
	likelyEmptyChroma.reserve(allChroma.size());
	for (const Features& feature: features) {
		if (!feature.valid) {
			continue;
		}
		const bool inBand     = std::abs(feature.deltaL - medianInitial) <= emptyBand;
		const bool lowSupport = (feature.darkFrac + feature.brightFrac) <= calibrationConfig.likelyEmptySupportSumMax;
		if (inBand && lowSupport) {
			likelyEmptyDelta.push_back(feature.deltaL);
			likelyEmptyChroma.push_back(feature.chromaSq);
		}
	}

	const std::size_t minEmptyCount = std::max<std::size_t>(
	        static_cast<std::size_t>(calibrationConfig.calibMinEmptySamples),
	        static_cast<std::size_t>(std::lround(calibrationConfig.calibMinEmptyFraction * static_cast<float>(boardSize) * static_cast<float>(boardSize))));

	float medianFinal = medianInitial;
	float sigmaFinal  = sigmaInitial;
	if (likelyEmptyDelta.size() >= minEmptyCount) {
		float medianLikely = 0.0f;
		float sigmaLikely  = 0.0f;
		if (robustMedianSigma(likelyEmptyDelta, calibrationConfig, medianLikely, sigmaLikely)) {
			medianFinal = medianLikely;
			sigmaFinal  = sigmaLikely;
		}
	}

	const std::vector<float>& chromaSource = likelyEmptyChroma.empty() ? allChroma : likelyEmptyChroma;
	float chromaMedian                     = calibrationConfig.chromaTFallback;
	if (!chromaSource.empty()) {
		std::vector<float> sortedChroma = chromaSource;
		std::sort(sortedChroma.begin(), sortedChroma.end());
		chromaMedian = medianSorted(sortedChroma);
	}

	outModel.medianEmpty = medianFinal;
	outModel.sigmaEmpty  = std::max(calibrationConfig.sigmaMin, sigmaFinal);
	outModel.tChromaSq   = std::max(calibrationConfig.chromaTMin, chromaMedian);
	return true;
}

} // namespace tengen::vision::core::ModelCalibration
