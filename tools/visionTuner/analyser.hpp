#pragma once

#include "pipelineStep.hpp"

#include <opencv2/core/mat.hpp>

namespace tengen::vision {

//! Runs the image detection pipeline with the DebugVisualizer attached to the desired PipelineStep.
class Analyser {
public:
	explicit Analyser(cv::Mat image);

	cv::Mat analyse(const PipelineStep step) const;

private:
	cv::Mat m_original;
};

} // namespace tengen::vision
