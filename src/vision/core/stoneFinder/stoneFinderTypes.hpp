#pragma once

#include "features.hpp"
#include "vision/core/stoneFinder.hpp"

#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace tengen::vision::core {

struct Radii {
	int innerRadius{0};
	int bgRadius{0};
	int bgOffset{0};
};

struct Offsets {
	std::vector<cv::Point> inner;
	std::vector<cv::Point> bg;
};

struct LabBlur {
	cv::Mat L;
	cv::Mat A;
	cv::Mat B;
};

struct SampleContext {
	const cv::Mat& L;
	const cv::Mat& A;
	const cv::Mat& B;
	int rows{0};
	int cols{0};
};

struct Model {
	float medianEmpty{0.0f};
	float sigmaEmpty{1.0f};
	float tChromaSq{0.0f};
};

struct Scores {
	float black{0.0f};
	float white{0.0f};
	float empty{0.0f};
	float chromaPenalty{0.0f};
	float z{0.0f};
};

struct Eval {
	StoneState state{StoneState::Empty};
	float bestScore{0.0f};
	float secondScore{0.0f};
	float margin{0.0f};
	float required{0.0f};
	float confidence{0.0f};
	float z{0.0f};
};

struct SpatialContext {
	int edgeLevel{0};
	float neighborMedian{0.0f};
	unsigned boardSize{0u};
};

enum class RejectionReason {
	None,
	WeakZ,
	LowConfidence,
	WeakSupport,
	WeakNeighborContrast,
	EdgeArtifact,
	MarginTooSmall,
	Other,
};

struct DebugStats {
	int blackCount{0};
	int whiteCount{0};
	int emptyCount{0};
	int refinedTried{0};
	int refinedAccepted{0};
};

} // namespace tengen::vision::core
