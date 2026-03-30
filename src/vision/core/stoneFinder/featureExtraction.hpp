#pragma once

#include "stoneFinderTypes.hpp"

namespace tengen::vision::core {
namespace FeatureExtraction {

bool convertToLab(const cv::Mat& image, cv::Mat& outLab);
bool prepareLabBlur(const cv::Mat& image, const Radii& radii, const GeometryConfig& config, LabBlur& outBlur);
bool sampleMeanL(const SampleContext& context, int centerX, int centerY, const std::vector<cv::Point>& offsets, float& outMean);
bool sampleMeanLab(const SampleContext& context, int centerX, int centerY, const std::vector<cv::Point>& offsets, float& outL, float& outA, float& outB);
bool computeFeaturesAt(const SampleContext& context, const Offsets& offsets, const Radii& radii, const GeometryConfig& config, int centerX, int centerY,
                       Features& outFeatures);
std::vector<Features> computeFeatures(const std::vector<cv::Point2f>& intersections, const SampleContext& context, const Offsets& offsets, const Radii& radii,
                                      const GeometryConfig& config);

} // namespace FeatureExtraction
} // namespace tengen::vision::core
