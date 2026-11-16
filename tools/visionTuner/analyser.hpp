#pragma once

#include "pipelineStep.hpp"

#include <opencv2/core/mat.hpp>
#include <opencv2/dnn.hpp>

namespace tengen::vision {

//! Mosaic of the debug images the pipeline produces up to step. stoneModel comes from core::loadStoneModel.
cv::Mat analyse(const cv::Mat& image, PipelineStep step, cv::dnn::Net& stoneModel);

} // namespace tengen::vision
