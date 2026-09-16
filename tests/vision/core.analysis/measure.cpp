#include "measure.hpp"

#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"

#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>

namespace tengen::vision::analysis {
namespace {

constexpr double TOLERANCE_FRACTION = 0.1; //!< Same corner tolerance the integration tests apply.

using Permutation = std::array<std::size_t, 4>;

Permutation bestPermutation(const Corners& reference, const Corners& other) {
	Permutation permutation{0u, 1u, 2u, 3u};
	Permutation best  = permutation;
	double bestTotal  = std::numeric_limits<double>::max();
	do {
		double total = 0.0;
		for (std::size_t i = 0u; i < reference.size(); ++i) {
			total += cv::norm(reference[i] - other[permutation[i]]);
		}
		if (total < bestTotal) {
			bestTotal = total;
			best      = permutation;
		}
	} while (std::next_permutation(permutation.begin(), permutation.end()));
	return best;
}

cv::Size2d boundingBoxSize(const Corners& corners) {
	float minX = corners[0].x;
	float maxX = corners[0].x;
	float minY = corners[0].y;
	float maxY = corners[0].y;
	for (const auto& corner: corners) {
		minX = std::min(minX, corner.x);
		maxX = std::max(maxX, corner.x);
		minY = std::min(minY, corner.y);
		maxY = std::max(maxY, corner.y);
	}
	return {static_cast<double>(maxX - minX), static_cast<double>(maxY - minY)};
}

double boundingBoxMinDimension(const Corners& corners) {
	const cv::Size2d size = boundingBoxSize(corners);
	return std::min(size.width, size.height);
}

Corners transformCorners(const Corners& corners, const cv::Mat& homography) {
	const std::vector<cv::Point2f> input(corners.begin(), corners.end());
	std::vector<cv::Point2f> output;
	cv::perspectiveTransform(input, output, homography);
	return {output[0], output[1], output[2], output[3]};
}

float medianOf(std::vector<float> values) {
	const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2u);
	std::nth_element(values.begin(), middle, values.end());
	return *middle;
}

} // namespace

double cornerDistance(const Corners& lhs, const Corners& rhs) {
	const Permutation permutation = bestPermutation(lhs, rhs);

	double worst = 0.0;
	for (std::size_t i = 0u; i < lhs.size(); ++i) {
		worst = std::max(worst, cv::norm(lhs[i] - rhs[permutation[i]]));
	}
	return worst;
}

Corners medianQuad(const std::vector<Corners>& quads) {
	Corners median{};
	if (quads.empty()) {
		return median;
	}

	std::array<std::vector<float>, 4> xs{};
	std::array<std::vector<float>, 4> ys{};
	for (const auto& quad: quads) {
		// Align to the first quad first: corner order is not guaranteed to agree across images.
		const Permutation permutation = bestPermutation(quads.front(), quad);
		for (std::size_t i = 0u; i < quad.size(); ++i) {
			xs[i].push_back(quad[permutation[i]].x);
			ys[i].push_back(quad[permutation[i]].y);
		}
	}

	for (std::size_t i = 0u; i < median.size(); ++i) {
		median[i] = {medianOf(xs[i]), medianOf(ys[i])};
	}
	return median;
}

Measurement measureImage(const std::filesystem::path& image, const std::optional<GroundTruth>& truth) {
	Measurement measurement{};
	measurement.image = image;

	const cv::Mat source = cv::imread(image.string());
	if (source.empty()) {
		return measurement;
	}

	const core::WarpResult warped = core::warpToBoard(source);
	if (!core::isValidBoard(warped)) {
		return measurement;
	}
	measurement.boardFound = true;
	measurement.contour    = warped.contourCorners;

	const core::BoardGeometry geometry = core::analyseGeometry(warped);
	if (core::isValidGeometry(geometry)) {
		measurement.boardSize = geometry.boardSize;
		measurement.spacing   = geometry.spacing;
	}

	if (!truth.has_value()) {
		return measurement;
	}

	measurement.errorToBoard = cornerDistance(truth->board, measurement.contour);
	measurement.errorToGrid  = cornerDistance(truth->grid, measurement.contour);
	if (measurement.errorToBoard <= TOLERANCE_FRACTION * boundingBoxMinDimension(truth->board)) {
		measurement.framing = Framing::Board;
	} else if (measurement.errorToGrid <= TOLERANCE_FRACTION * boundingBoxMinDimension(truth->grid)) {
		measurement.framing = Framing::Grid;
	}

	const cv::Size2d fill = boundingBoxSize(transformCorners(truth->grid, warped.H0));
	measurement.gridFill  = 0.5 * (fill.width / warped.imageB0.cols + fill.height / warped.imageB0.rows);

	if (measurement.boardSize >= 2u) {
		const double expected = boundingBoxMinDimension(transformCorners(truth->grid, geometry.H)) / static_cast<double>(measurement.boardSize - 1u);
		measurement.spacingError = (expected > 0.0) ? (measurement.spacing - expected) / expected : 0.0;
	}

	return measurement;
}

} // namespace tengen::vision::analysis
