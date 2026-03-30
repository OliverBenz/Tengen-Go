#include "vision/core/boardFinder.hpp"

#include "boardFinderInternal.hpp"

namespace tengen::vision::core {

//! Find the board in an image and crop/scale/rectify so the image is of a planar board.
WarpResult warpToBoard(const cv::Mat& image, DebugVisualizer* debugger) {
	const auto fail = [&](const std::string& message) -> WarpResult {
		std::cerr << message << '\n';
		if (debugger) {
			debugger->endStage();
		}
		return {};
	};

	if (debugger) {
		debugger->beginStage("Warp To Board");
		debugger->add("Input", image);
	}

	if (image.empty()) {
		return fail("Failed to load image");
	}

	cv::Mat gray;
	if (!internal::convertToGray(image, gray)) {
		return fail("Unsupported input channel count");
	}
	if (debugger)
		debugger->add("Grayscale", gray);

	const auto settings = internal::choosePreprocessSettings(image.size());

	cv::Mat blurred;
	cv::GaussianBlur(gray, blurred, cv::Size(settings.blurKernelSize, settings.blurKernelSize), 1.5);
	if (debugger)
		debugger->add("Gaussian Blur", blurred);

	const auto masks = internal::buildCandidateMasks(blurred, settings);
	if (debugger) {
		debugger->add("Edge Mask", masks.edgeMask);
		debugger->add("Bright Mask", masks.brightMask);
		debugger->add("Dark Mask", masks.darkMask);
	}

	std::vector<std::vector<cv::Point>> contours;
	std::vector<std::vector<cv::Point>> contoursExternal;
	internal::appendContours(masks.edgeMask, contours, &contoursExternal);
	internal::appendContours(masks.brightMask, contours, &contoursExternal);
	internal::appendContours(masks.darkMask, contours, &contoursExternal);
	if (contours.empty()) {
		return fail("No contours found");
	}

	if (debugger) {
		cv::Mat drawnContours = image.clone();
		cv::drawContours(drawnContours, contoursExternal, -1, cv::Scalar(255, 0, 0), 2);
		debugger->add("Contour Finder", drawnContours);
	}

	const auto bestCandidate = internal::selectBestBoardCandidate(contours, image);
	if (!bestCandidate.has_value()) {
		return fail("No valid board candidate found");
	}

	DEBUG_LOG("Largest contour idx: " << bestCandidate->contourIdx << " area: " << bestCandidate->area << "\n");
	if (debugger) {
		cv::Mat selected = image.clone();
		std::vector<cv::Point> poly;
		poly.reserve(bestCandidate->quad.size());
		for (const auto& p: bestCandidate->quad) {
			poly.emplace_back(static_cast<int>(std::lround(p.x)), static_cast<int>(std::lround(p.y)));
		}
		cv::polylines(selected, poly, true, cv::Scalar(0, 255, 0), 3);
		debugger->add("Contour Selected", selected);
	}

	const std::vector<cv::Point2f> dst = {
	        {0.f, 0.f},
	        {static_cast<float>(internal::WARP_OUT_SIZE) - 1.f, 0.f},
	        {static_cast<float>(internal::WARP_OUT_SIZE) - 1.f, static_cast<float>(internal::WARP_OUT_SIZE) - 1.f},
	        {0.f, static_cast<float>(internal::WARP_OUT_SIZE) - 1.f},
	};

	cv::Mat H = cv::getPerspectiveTransform(bestCandidate->quad, dst);

	cv::Mat warped;
	cv::warpPerspective(image, warped, H, cv::Size(internal::WARP_OUT_SIZE, internal::WARP_OUT_SIZE));
	if (bestCandidate->fromRelaxedPass) {
		warped = internal::enhanceWarpContrast(warped);
	}
	if (debugger) {
		debugger->add("Warped", warped);
		debugger->endStage();
	}

	return {warped, H};
}

bool isValidBoard(const WarpResult& board) {
	const auto validWarped     = !board.imageB0.empty(); // TODO: Additional checks.
	const auto validHomography = !board.H0.empty();      // TODO: Additional checks.

	return validWarped && validHomography;
}


} // namespace tengen::vision::core
