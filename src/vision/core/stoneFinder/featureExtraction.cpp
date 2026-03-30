#include "featureExtraction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace tengen::vision::core {
namespace FeatureExtraction {

bool convertToLab(const cv::Mat& image, cv::Mat& outLab) {
	if (image.channels() == 3) {
		cv::cvtColor(image, outLab, cv::COLOR_BGR2Lab);
		return true;
	}

	if (image.channels() == 4) {
		cv::Mat bgrImage;
		cv::cvtColor(image, bgrImage, cv::COLOR_BGRA2BGR);
		cv::cvtColor(bgrImage, outLab, cv::COLOR_BGR2Lab);
		return true;
	}

	if (image.channels() == 1) {
		cv::Mat bgrImage;
		cv::cvtColor(image, bgrImage, cv::COLOR_GRAY2BGR);
		cv::cvtColor(bgrImage, outLab, cv::COLOR_BGR2Lab);
		return true;
	}

	return false;
}

bool prepareLabBlur(const cv::Mat& image, const Radii& radii, const GeometryConfig& config, LabBlur& outBlur) {
	cv::Mat labImage;
	if (!convertToLab(image, labImage)) {
		return false;
	}

	cv::extractChannel(labImage, outBlur.L, 0);
	cv::extractChannel(labImage, outBlur.A, 1);
	cv::extractChannel(labImage, outBlur.B, 2);

	const double sigma = std::clamp(config.blurSigmaRadiusK * static_cast<double>(radii.innerRadius), config.blurSigmaMin, config.blurSigmaMax);

	cv::GaussianBlur(outBlur.L, outBlur.L, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
	cv::GaussianBlur(outBlur.A, outBlur.A, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
	cv::GaussianBlur(outBlur.B, outBlur.B, cv::Size(), sigma, sigma, cv::BORDER_REPLICATE);
	return true;
}

bool sampleMeanL(const SampleContext& context, int centerX, int centerY, const std::vector<cv::Point>& offsets, float& outMean) {
	std::uint32_t sum   = 0u;
	std::uint32_t count = 0u;
	for (const cv::Point& offset: offsets) {
		const int x = centerX + offset.x;
		const int y = centerY + offset.y;
		if (x < 0 || x >= context.cols || y < 0 || y >= context.rows) {
			continue;
		}
		sum += static_cast<std::uint32_t>(context.L.ptr<std::uint8_t>(y)[x]);
		++count;
	}
	if (count == 0u) {
		return false;
	}
	outMean = static_cast<float>(sum) / static_cast<float>(count);
	return true;
}

bool sampleMeanLab(const SampleContext& context, int centerX, int centerY, const std::vector<cv::Point>& offsets, float& outL, float& outA, float& outB) {
	std::uint32_t sumL  = 0u;
	std::uint32_t sumA  = 0u;
	std::uint32_t sumB  = 0u;
	std::uint32_t count = 0u;
	for (const cv::Point& offset: offsets) {
		const int x = centerX + offset.x;
		const int y = centerY + offset.y;
		if (x < 0 || x >= context.cols || y < 0 || y >= context.rows) {
			continue;
		}
		sumL += static_cast<std::uint32_t>(context.L.ptr<std::uint8_t>(y)[x]);
		sumA += static_cast<std::uint32_t>(context.A.ptr<std::uint8_t>(y)[x]);
		sumB += static_cast<std::uint32_t>(context.B.ptr<std::uint8_t>(y)[x]);
		++count;
	}
	if (count == 0u) {
		return false;
	}

	outL = static_cast<float>(sumL) / static_cast<float>(count);
	outA = static_cast<float>(sumA) / static_cast<float>(count);
	outB = static_cast<float>(sumB) / static_cast<float>(count);
	return true;
}

bool computeFeaturesAt(const SampleContext& context, const Offsets& offsets, const Radii& radii, const GeometryConfig& config, int centerX, int centerY,
                       Features& outFeatures) {
	outFeatures = Features{};

	float innerL = 0.0f;
	float innerA = 0.0f;
	float innerB = 0.0f;
	if (!sampleMeanLab(context, centerX, centerY, offsets.inner, innerL, innerA, innerB)) {
		return false;
	}

	std::array<float, 8> backgroundSamples{};
	int backgroundCount  = 0;
	float backgroundMean = 0.0f;
	const std::array<cv::Point, 8> directions{
	        cv::Point(-1, -1), cv::Point(1, -1), cv::Point(-1, 1), cv::Point(1, 1), cv::Point(-1, 0), cv::Point(1, 0), cv::Point(0, -1), cv::Point(0, 1),
	};
	for (const cv::Point& direction: directions) {
		const int sampleX = centerX + direction.x * radii.bgOffset;
		const int sampleY = centerY + direction.y * radii.bgOffset;
		if (sampleMeanL(context, sampleX, sampleY, offsets.bg, backgroundMean)) {
			backgroundSamples[static_cast<std::size_t>(backgroundCount++)] = backgroundMean;
		}
	}
	if (backgroundCount < config.minBgSamples) {
		return false;
	}

	std::sort(backgroundSamples.begin(), backgroundSamples.begin() + backgroundCount);
	const float backgroundMedian = (backgroundCount % 2 == 1) ? backgroundSamples[static_cast<std::size_t>(backgroundCount / 2)]
	                                                          : 0.5f * (backgroundSamples[static_cast<std::size_t>(backgroundCount / 2 - 1)] +
	                                                                    backgroundSamples[static_cast<std::size_t>(backgroundCount / 2)]);

	outFeatures.deltaL   = innerL - backgroundMedian;
	const float deltaA   = innerA - config.labNeutral;
	const float deltaB   = innerB - config.labNeutral;
	outFeatures.chromaSq = deltaA * deltaA + deltaB * deltaB;

	std::uint32_t total  = 0u;
	std::uint32_t dark   = 0u;
	std::uint32_t bright = 0u;
	for (const cv::Point& offset: offsets.inner) {
		const int x = centerX + offset.x;
		const int y = centerY + offset.y;
		if (x < 0 || x >= context.cols || y < 0 || y >= context.rows) {
			continue;
		}
		const float diff = static_cast<float>(context.L.ptr<std::uint8_t>(y)[x]) - backgroundMedian;
		++total;
		if (diff <= -config.supportDelta) {
			++dark;
		} else if (diff >= config.supportDelta) {
			++bright;
		}
	}

	if (total > 0u) {
		outFeatures.darkFrac   = static_cast<float>(dark) / static_cast<float>(total);
		outFeatures.brightFrac = static_cast<float>(bright) / static_cast<float>(total);
	}

	outFeatures.valid = true;
	return true;
}

std::vector<Features> computeFeatures(const std::vector<cv::Point2f>& intersections, const SampleContext& context, const Offsets& offsets, const Radii& radii,
                                      const GeometryConfig& config) {
	std::vector<Features> allFeatures(intersections.size());
	for (std::size_t index = 0; index < intersections.size(); ++index) {
		const int centerX = static_cast<int>(std::lround(intersections[index].x));
		const int centerY = static_cast<int>(std::lround(intersections[index].y));
		computeFeaturesAt(context, offsets, radii, config, centerX, centerY, allFeatures[index]);
	}
	return allFeatures;
}

} // namespace FeatureExtraction
} // namespace tengen::vision::core
