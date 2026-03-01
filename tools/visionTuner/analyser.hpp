#pragma once

#include "pipelineStep.hpp"

#include <opencv2/core/mat.hpp>

namespace tengen::vision {

//! Runs the image detection pipeline with the DebugVisualizer attached to the desired PipelineStep.
class Analyser {
public:
	cv::Mat analyse(const cv::Mat& image, PipelineStep step) const;
};

} // namespace tengen::vision
