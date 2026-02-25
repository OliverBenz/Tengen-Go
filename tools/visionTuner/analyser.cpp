#include "analyser.hpp"

#include "vision/core/boardFinder.hpp"
#include "vision/core/debugVisualizer.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>

namespace tengen::vision {


static cv::Mat buildInfoTile(const std::string& title, const std::string& message) {
	cv::Mat tile(540, 960, CV_8UC3, cv::Scalar(20, 20, 20));
	cv::putText(tile, title, cv::Point(40, 120), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(250, 250, 250), 2, cv::LINE_AA);
	cv::putText(tile, message, cv::Point(40, 200), cv::FONT_HERSHEY_SIMPLEX, 0.85, cv::Scalar(200, 200, 200), 2, cv::LINE_AA);
	return tile;
}


cv::Mat Analyser::analyse(const cv::Mat& image, const PipelineStep step) const {
	if (image.empty()) {
		return buildInfoTile("Input Error", "Could not load image.");
	}

	core::DebugVisualizer debugger;
	debugger.setInteractive(false);

	switch (step) {
	case tengen::PipelineStep::FindBoard:
		core::warpToBoard(image, &debugger);
		break;

	case tengen::PipelineStep::ConstructGeometry: {
		const core::WarpResult board = core::warpToBoard(image);
		if (!core::isValidBoard(board)) {
			return buildInfoTile("Construct Geometry", "warpToBoard failed for this image.");
		}
		core::rectifyImage(image, board, &debugger);
		break;
	}

	case tengen::PipelineStep::FindStones: {
		const core::WarpResult board = core::warpToBoard(image);
		if (!core::isValidBoard(board)) {
			return buildInfoTile("Find Stones", "warpToBoard failed for this image.");
		}

		const core::BoardGeometry geometry = core::rectifyImage(image, board);
		if (!core::isValidGeometry(geometry)) {
			return buildInfoTile("Find Stones", "rectifyImage failed for this image.");
		}

		const core::StoneResult result = core::analyseBoard(geometry, &debugger);
		if (!result.success) {
			return buildInfoTile("Find Stones", "analyseBoard failed for this image.");
		}
		break;
	}

	case tengen::PipelineStep::All: {
		const core::WarpResult board = core::warpToBoard(image, &debugger);
		if (!core::isValidBoard(board)) {
			break;
		}

		const core::BoardGeometry geometry = core::rectifyImage(image, board, &debugger);
		if (!core::isValidGeometry(geometry)) {
			break;
		}

		core::analyseBoard(geometry, &debugger);
		break;
	}
	}

	const cv::Mat mosaic = debugger.buildMosaic();
	if (mosaic.empty()) {
		return buildInfoTile("No Debug Output", "Selected stage produced no visuals.");
	}
	return mosaic;
}


} // namespace tengen::vision
