#include "report.hpp"

#include <algorithm>
#include <format>
#include <iostream>

namespace tengen::vision::analysis {
namespace {

// A frame is flagged against its own set's typical drift, since not every set was shot from a tripod.
constexpr double DRIFT_FLOOR    = 5.0; //!< Below this a frame is never flagged (px).
constexpr double OUTLIER_FACTOR = 5.0; //!< Multiple of the set's median drift that counts as an outlier.

double medianOf(std::vector<double> values) {
	if (values.empty()) {
		return 0.0;
	}
	const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2u);
	std::nth_element(values.begin(), middle, values.end());
	return *middle;
}

std::string framingName(const Framing framing) {
	switch (framing) {
	case Framing::Board:
		return "board";
	case Framing::Grid:
		return "grid";
	case Framing::None:
		break;
	}
	return "NEITHER";
}

std::string boardSizeText(const Measurement& measurement) {
	return measurement.boardSize == 0u ? "-" : std::to_string(measurement.boardSize);
}

std::string quadText(const Corners& quad) {
	std::string text;
	for (const auto& corner: quad) {
		text += std::format(" ({:.0f},{:.0f})", corner.x, corner.y);
	}
	return text;
}

void printHeader(const ImageSet& set, const std::string_view kind) {
	std::cout << std::format("\n== {} ({} images, {}) ==\n", set.name, set.images.size(), kind);
}

std::size_t countSized(const std::vector<Measurement>& measurements) {
	return static_cast<std::size_t>(std::count_if(measurements.begin(), measurements.end(), [](const Measurement& m) { return m.boardSize != 0u; }));
}

} // namespace

void printLabelledReport(const ImageSet& set, const std::vector<Measurement>& measurements) {
	printHeader(set, "labelled");
	std::cout << std::format("  {:<16} {:>3}  {:<8} {:>8} {:>8} {:>9} {:>11}\n", "image", "N", "framing", "errBoard", "errGrid", "gridFill", "spacingErr");

	std::size_t framedOnBoard = 0u;
	std::size_t framedOnGrid  = 0u;
	double worstSpacingError  = 0.0;

	for (const auto& measurement: measurements) {
		const std::string name = measurement.image.filename().string();
		if (!measurement.boardFound) {
			std::cout << std::format("  {:<16} {:>3}  {}\n", name, "-", "BOARDFINDER FAILED");
			continue;
		}

		framedOnBoard += (measurement.framing == Framing::Board) ? 1u : 0u;
		framedOnGrid += (measurement.framing == Framing::Grid) ? 1u : 0u;
		worstSpacingError = std::max(worstSpacingError, std::abs(measurement.spacingError));

		std::cout << std::format("  {:<16} {:>3}  {:<8} {:>8.2f} {:>8.2f} {:>9.3f} {:>10.2f}%\n", name, boardSizeText(measurement),
		                         framingName(measurement.framing), measurement.errorToBoard, measurement.errorToGrid, measurement.gridFill,
		                         100.0 * measurement.spacingError);
	}

	const std::size_t none = measurements.size() - framedOnBoard - framedOnGrid;
	std::cout << std::format("  -> framing: board {}, grid {}, neither {} | sized {}/{} | worst spacingErr {:.2f}%\n", framedOnBoard, framedOnGrid, none,
	                         countSized(measurements), measurements.size(), 100.0 * worstSpacingError);
}

void printSeriesReport(const ImageSet& set, const std::vector<Measurement>& measurements) {
	printHeader(set, "fixed-camera series");

	std::vector<Corners> quads;
	for (const auto& measurement: measurements) {
		if (measurement.boardFound) {
			quads.push_back(measurement.contour);
		}
	}
	if (quads.empty()) {
		std::cout << "  no image produced a board\n";
		return;
	}

	const Corners reference = medianQuad(quads);

	std::vector<double> drifts;
	drifts.reserve(quads.size());
	for (const auto& quad: quads) {
		drifts.push_back(cornerDistance(reference, quad));
	}
	const double threshold = std::max(DRIFT_FLOOR, OUTLIER_FACTOR * medianOf(drifts));

	std::cout << std::format("  reference quad:{}\n", quadText(reference));
	std::cout << std::format("  {:<16} {:>3} {:>9}\n", "image", "N", "drift");

	double worstDrift    = 0.0;
	std::size_t outliers = 0u;

	for (const auto& measurement: measurements) {
		const std::string name = measurement.image.filename().string();
		if (!measurement.boardFound) {
			std::cout << std::format("  {:<16} {:>3}  {}\n", name, "-", "BOARDFINDER FAILED");
			continue;
		}

		const double drift   = cornerDistance(reference, measurement.contour);
		const bool isOutlier = drift > threshold;
		worstDrift           = std::max(worstDrift, drift);
		outliers += isOutlier ? 1u : 0u;

		std::cout << std::format("  {:<16} {:>3} {:>9.2f}{}\n", name, boardSizeText(measurement), drift, isOutlier ? "  <-- moved" : "");
	}

	std::cout << std::format("  -> sized {}/{} | max drift {:.2f}px | {} outlier(s) above {:.2f}px\n", countSized(measurements), measurements.size(),
	                         worstDrift, outliers, threshold);
}

} // namespace tengen::vision::analysis
