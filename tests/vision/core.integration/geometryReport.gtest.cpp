#include "geometryGroundTruth.hpp"
#include "testDataHelpers.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

// Measurement baseline for the two geometry stages (see src/vision/core/README.md for the space
// terminology). Process.* and Rectifier.* assert pass/fail on the pipeline as a whole, which hides two
// things a stage rewrite needs to see: how much margin an image still has before it fails, and which
// stage caused a failure. This reports both, continuously, over every image that has manually labelled
// geometry ground truth - so a BoardFinder/GridFinder change can be compared corpus-wide instead of
// only on the image that motivated it.
namespace tengen::vision::core {
namespace gtest {
namespace {

//! Accuracy measured for one image. Non-finite metrics could not be computed (stage failed earlier).
struct GeometryMeasurement {
	std::string set;                   //!< Resource directory name.
	std::string image;                 //!< Image file name.
	unsigned expectedBoardSize{0u};    //!< Board size from the ground truth.
	unsigned detectedBoardSize{0u};    //!< Board size reported by GridFinder (0 if it failed).
	double iouBoard{0.0};              //!< IoU of the BoardFinder quad against the physical board outline.
	double iouGrid{0.0};               //!< IoU of the BoardFinder quad against the outermost grid lines.
	double gridFill{0.0};              //!< Fraction of the B_0 canvas the true grid actually spans (see below).
	double stage1CornerErrFrac{0.0};   //!< Max BoardFinder corner error / B_0 canvas min-dimension.
	double stage2GridErrSpacing{0.0};  //!< Max GridFinder corner error / detected grid spacing.
	std::string status;                //!< ok | offTol | wrongN | gridFail | warpFail | loadFail.
};

//! Tolerances Process.Board_Detect_Easy asserts, repeated here so the report shows remaining margin.
constexpr double STAGE1_ASSERTED_TOLERANCE = 0.10;
constexpr double STAGE2_ASSERTED_TOLERANCE = 0.30;

constexpr double NOT_MEASURED = std::numeric_limits<double>::quiet_NaN();

//! Run both geometry stages on one image and measure them against its ground truth.
//! \note Deliberately does not reuse runPipeline(): that asserts on intermediate steps, which would turn
//!       every already-known failure into gtest noise and abort the sweep.
GeometryMeasurement measureImage(const std::filesystem::path& imagePath, std::string set, std::string image) {
	GeometryMeasurement measurement{std::move(set), std::move(image)};
	measurement.iouBoard             = NOT_MEASURED;
	measurement.iouGrid              = NOT_MEASURED;
	measurement.gridFill             = NOT_MEASURED;
	measurement.stage1CornerErrFrac  = NOT_MEASURED;
	measurement.stage2GridErrSpacing = NOT_MEASURED;

	const cv::Mat source = cv::imread(imagePath.string());
	if (source.empty()) {
		measurement.status = "loadFail";
		return measurement;
	}

	const GeometryGroundTruth truth = loadGeometryGroundTruth(imagePath);
	measurement.expectedBoardSize   = truth.boardSize;

	// Stage 1: BoardFinder (H_0: I_B -> B_0).
	const WarpResult warped = warpToBoard(source);
	if (!isValidBoard(warped)) {
		measurement.status = "warpFail";
		return measurement;
	}

	const double canvasWidth                     = static_cast<double>(warped.imageB0.cols);
	const double canvasHeight                    = static_cast<double>(warped.imageB0.rows);
	const std::vector<cv::Point2f> canvasCorners = {
	        {0.f, 0.f},
	        {static_cast<float>(canvasWidth - 1.0), 0.f},
	        {static_cast<float>(canvasWidth - 1.0), static_cast<float>(canvasHeight - 1.0)},
	        {0.f, static_cast<float>(canvasHeight - 1.0)},
	};

	// IoU is measured back in original-image space: it is resolution independent, and it stays
	// comparable if the stage-1 output contract ever changes from the board outline to the grid
	// outline (a grid-first BoardFinder would naturally produce the latter), because both are reported.
	std::vector<cv::Point2f> detectedQuad;
	cv::perspectiveTransform(canvasCorners, detectedQuad, warped.H0.inv());
	measurement.iouBoard = quadIoU(detectedQuad, truth.boardCorners);
	measurement.iouGrid  = quadIoU(detectedQuad, truth.gridCorners);

	// Same quantity Process.Board_Detect_Easy asserts on, reported as a number instead of pass/fail.
	std::vector<cv::Point2f> boardCornersWarped;
	cv::perspectiveTransform(truth.boardCorners, boardCornersWarped, warped.H0);
	measurement.stage1CornerErrFrac = maxMatchedPointDistance(canvasCorners, boardCornersWarped) / std::min(canvasWidth, canvasHeight);

	// How much of the B_0 canvas the true grid really spans. GridFinder passes the full canvas size to
	// findGrid() as the expected grid extent, i.e. it assumes this is 1.0; whatever it actually is here
	// is the error in that assumption, and findGrid() ranks board sizes by extent fit first.
	std::vector<cv::Point2f> gridCornersB0;
	cv::perspectiveTransform(truth.gridCorners, gridCornersB0, warped.H0);
	const auto [minX, maxX] = std::ranges::minmax(gridCornersB0 | std::views::transform(&cv::Point2f::x));
	const auto [minY, maxY] = std::ranges::minmax(gridCornersB0 | std::views::transform(&cv::Point2f::y));
	measurement.gridFill    = 0.5 * ((maxX - minX) / canvasWidth + (maxY - minY) / canvasHeight);

	// Stage 2: GridFinder (H: I_B -> B).
	const BoardGeometry geometry = analyseGeometry(warped);
	if (!isValidGeometry(geometry)) {
		measurement.status = "gridFail";
		return measurement;
	}
	measurement.detectedBoardSize = geometry.boardSize;

	if (geometry.spacing > 0.0) {
		std::vector<cv::Point2f> gridCornersWarped;
		cv::perspectiveTransform(truth.gridCorners, gridCornersWarped, geometry.H);

		const auto n                                       = geometry.boardSize;
		const auto& intersections                          = geometry.intersections; // isValidGeometry() guarantees n*n entries.
		const std::vector<cv::Point2f> detectedGridCorners = {
		        intersections[0],
		        intersections[n - 1u],
		        intersections[(n - 1u) * n],
		        intersections[n * n - 1u],
		};
		measurement.stage2GridErrSpacing = maxMatchedPointDistance(detectedGridCorners, gridCornersWarped) / geometry.spacing;
	}

	if (measurement.detectedBoardSize != measurement.expectedBoardSize) {
		measurement.status = "wrongN";
		return measurement;
	}

	// A correct board size does not imply correct geometry: an image can report N=13 and still place the
	// grid many spacings away from where it actually is. Separate those from genuinely clean results.
	const bool withinTolerance =
	        measurement.stage1CornerErrFrac <= STAGE1_ASSERTED_TOLERANCE && measurement.stage2GridErrSpacing <= STAGE2_ASSERTED_TOLERANCE;
	measurement.status = withinTolerance ? "ok" : "offTol";
	return measurement;
}

//! Format a metric, marking values that could not be measured.
std::string formatMetric(const double value) {
	return std::isfinite(value) ? std::format("{:.3f}", value) : "-";
}

//! Median and the value furthest in the failing direction, over the measurable entries.
std::string summarize(std::vector<double> values, const bool higherIsBetter) {
	std::erase_if(values, [](const double v) { return !std::isfinite(v); });
	if (values.empty()) {
		return "n/a";
	}

	std::sort(values.begin(), values.end());
	const double median = values[values.size() / 2u];
	const double worst  = higherIsBetter ? values.front() : values.back();
	return std::format("median={:.3f} worst={:.3f} (n={})", median, worst, values.size());
}

//! Every image that ships a .json geometry ground truth, grouped by resource directory.
std::vector<std::pair<std::string, std::vector<std::string>>> labelledImageSets() {
	std::vector<std::string> angleImages;
	for (unsigned i = 1u; i <= 6u; ++i) {
		angleImages.push_back(std::format("angle_{}.jpeg", i));
	}

	std::vector<std::string> sizeImages;
	for (const unsigned size: {9u, 13u, 19u}) {
		sizeImages.push_back(std::format("size_{}.jpeg", size));
	}

	return {
	        {"angled_easy", angleImages},
	        {"angled_hard", angleImages},
	        {"empty_angle_none", sizeImages},
	        {"empty_angle_small", sizeImages},
	};
}

} // namespace

// Reports, and deliberately does not assert on, stage accuracy: regressions are gated by Process.* and
// Rectifier.*, while this exists to show how those verdicts were reached and how close each image is to
// flipping. Diff the CSV block between two builds to compare a pipeline change corpus-wide:
//   visionCore.integration --gtest_filter=GeometryReport.* | grep ^CSV, > before.csv
TEST(GeometryReport, StageAccuracy) {
	std::vector<GeometryMeasurement> measurements;
	for (const auto& [set, images]: labelledImageSets()) {
		for (const auto& image: images) {
			measurements.push_back(measureImage(std::filesystem::path(PATH_TEST_IMG) / set / image, set, image));
		}
	}
	ASSERT_FALSE(measurements.empty());

	std::cout << "\n=== Geometry report: stage accuracy vs. manually labelled ground truth ===\n"
	          << "  iouBoard  BoardFinder quad vs. physical board outline   (higher is better)\n"
	          << "  iouGrid   BoardFinder quad vs. outermost grid lines     (higher is better)\n"
	          << "  gridFill  fraction of the B_0 canvas the true grid spans (GridFinder assumes 1.00)\n"
	          << std::format("  stage1Err max board-corner error / B_0 canvas min-dim  (Process asserts <= {:.2f})\n", STAGE1_ASSERTED_TOLERANCE)
	          << std::format("  stage2Err max grid-corner error / grid spacing         (Process asserts <= {:.2f})\n\n", STAGE2_ASSERTED_TOLERANCE);

	std::cout << std::format("{:<18}{:<12}{:>3}{:>5}{:>10}{:>9}{:>10}{:>11}{:>11}  {}\n", "set", "image", "N", "det", "iouBoard", "iouGrid", "gridFill",
	                         "stage1Err", "stage2Err", "status");
	for (const auto& m: measurements) {
		std::cout << std::format("{:<18}{:<12}{:>3}{:>5}{:>10}{:>9}{:>10}{:>11}{:>11}  {}\n", m.set, m.image, m.expectedBoardSize, m.detectedBoardSize,
		                         formatMetric(m.iouBoard), formatMetric(m.iouGrid), formatMetric(m.gridFill), formatMetric(m.stage1CornerErrFrac),
		                         formatMetric(m.stage2GridErrSpacing), m.status);
	}

	const auto column = [&measurements](double GeometryMeasurement::* field) {
		std::vector<double> values;
		values.reserve(measurements.size());
		for (const auto& m: measurements) {
			values.push_back(m.*field);
		}
		return values;
	};
	const auto countStatus = [&measurements](const std::string_view status) {
		return std::count_if(measurements.begin(), measurements.end(), [status](const GeometryMeasurement& m) { return m.status == status; });
	};

	std::cout << std::format("\n  images    {}\n", measurements.size())
	          << std::format("  ok        {}   offTol {}   wrongN {}   gridFail {}   warpFail {}   loadFail {}\n", countStatus("ok"), countStatus("offTol"),
	                         countStatus("wrongN"), countStatus("gridFail"), countStatus("warpFail"), countStatus("loadFail"))
	          << std::format("  iouBoard  {}\n", summarize(column(&GeometryMeasurement::iouBoard), /*higherIsBetter=*/true))
	          << std::format("  iouGrid   {}\n", summarize(column(&GeometryMeasurement::iouGrid), /*higherIsBetter=*/true))
	          << std::format("  gridFill  {}\n", summarize(column(&GeometryMeasurement::gridFill), /*higherIsBetter=*/true))
	          << std::format("  stage1Err {}\n", summarize(column(&GeometryMeasurement::stage1CornerErrFrac), /*higherIsBetter=*/false))
	          << std::format("  stage2Err {}\n", summarize(column(&GeometryMeasurement::stage2GridErrSpacing), /*higherIsBetter=*/false));

	std::cout << "\nCSV,set,image,expectedN,detectedN,iouBoard,iouGrid,gridFill,stage1Err,stage2Err,status\n";
	for (const auto& m: measurements) {
		std::cout << std::format("CSV,{},{},{},{},{},{},{},{},{},{}\n", m.set, m.image, m.expectedBoardSize, m.detectedBoardSize, formatMetric(m.iouBoard),
		                         formatMetric(m.iouGrid), formatMetric(m.gridFill), formatMetric(m.stage1CornerErrFrac),
		                         formatMetric(m.stage2GridErrSpacing), m.status);
	}
	std::cout << std::endl;
}

} // namespace gtest
} // namespace tengen::vision::core
