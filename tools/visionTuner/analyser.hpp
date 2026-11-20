#pragma once

#include "pipelineStep.hpp"

#include "vision/core/stoneClassifier.hpp"

#include <opencv2/core/mat.hpp>

namespace tengen::vision {

cv::Mat analyse(const cv::Mat& image, PipelineStep step, core::StoneClassifier& classifier);

} // namespace tengen::vision
