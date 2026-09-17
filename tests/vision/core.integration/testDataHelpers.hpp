#pragma once

#include "model/board.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"

#include <array>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string_view>
#include <vector>

namespace tengen::vision::core {
namespace gtest {

//! Output of every stage of the vision pipeline.
struct PipelineResult {
	WarpResult warped;        //!< BoardFinder stage: image warped onto the rough board contour.
	RectifiedBoard rectified; //!< GridFinder stage: rectified board image and its geometry.
	StoneResult stoneStep;    //!< StoneFinder stage: stone state per grid intersection.
};

std::vector<std::filesystem::path> getImagesInDirectory(const std::filesystem::path& directory); //!< Get all image files in a directory, ordered by name.
void ensureJsonExists(const std::vector<std::filesystem::path>& images);                         //!< Verify the json file for each image exists.

//! Warp an image onto the given board corners, producing what an ideal BoardFinder stage would.
WarpResult prepareIdealImage(const cv::Mat& image, const std::array<cv::Point2f, 4>& corners);

PipelineResult runPipeline(const cv::Mat& image);                           //!< Run all pipeline stages: board, grid and stone detection.
PipelineResult runPipeline(const cv::Mat& image, const WarpResult& warped); //!< Run the grid and stone detection stages on a given board warp.
bool isValidPipelineResult(const PipelineResult& result);                   //!< True if every pipeline stage produced a usable result.

//! Load the dotBW stone layout of an image: "<image>.txt", or the test sets "board.txt" if all its images show one and the same board.
Board loadExpectedBoard(const std::filesystem::path& imagePath);

//! Check every board coordinate against a ground truth board.
//! \note A photo does not tell us from which side the board was taken, so the layout is matched in all four rotations.
void expectStonesMatchBoard(const std::vector<StoneState>& stones, const Board& expected, std::string_view context);

} // namespace gtest
} // namespace tengen::vision::core
