#include "vision/vision.hpp"
#include "include/vision/vision.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>


namespace tengen::vision {

Vision::Vision(Source source) : m_source{source} {
}

Vision::~Vision() {
	disconnect();
	stop();
}


bool Vision::setup(const Coord gaugeCoord) {
	const auto stoneCount = [](const std::vector<core::StoneState>& input) -> std::size_t {
		return static_cast<std::size_t>(
		        std::count_if(input.begin(), input.end(), [](core::StoneState s) { return s == core::StoneState::Black || s == core::StoneState::White; }));
	};

	// Get image from source
	cv::Mat image;
	switch (m_source) {
	case Source::None:
		return false;
	case Source::Image:
		if (m_setupImagePath.empty()) {
			assert(false); // Set the image path before calling setup.
			return false;
		}
		image = cv::imread(m_setupImagePath.string(), cv::IMREAD_COLOR);
		break;
	case Source::Video: {
		cv::VideoCapture capture{0, cv::CAP_ANY};
		if (!capture.isOpened() || !capture.read(image) || image.empty()) {
			return false;
		}
		break;
	}
	}
	if (image.empty()) {
		return false;
	}

	const core::WarpResult warped = core::warpToBoard(image);
	if (!core::isValidBoard(warped)) {
		// TODO: Log
		return false;
	}

	core::BoardGeometry geometry = core::analyseGeometry(warped);
	core::transformImage(image, geometry);
	if (!core::isValidGeometry(geometry)) {
		// TODO: Log
		return false;
	}

	const core::StoneResult result = core::analyseBoard(geometry);
	if (!result.success || stoneCount(result.stones) != 1) {
		// TODO: Log
		return false;
	}

	const unsigned boardSize = geometry.boardSize;
	// TODO: Or gauge stone in center
	if (gaugeCoord.x >= boardSize || gaugeCoord.y >= boardSize) {
		return false;
	}

	// Get Stone Coordinates
	const auto stoneIt = std::find_if(result.stones.begin(), result.stones.end(),
	                                  [](core::StoneState s) { return s == core::StoneState::Black || s == core::StoneState::White; });
	if (stoneIt == result.stones.end()) {
		return false;
	}
	const std::size_t stoneIndex = static_cast<std::size_t>(std::distance(result.stones.begin(), stoneIt));
	const Coord placedCoord{
	        static_cast<unsigned>(stoneIndex / boardSize),
	        static_cast<unsigned>(stoneIndex % boardSize),
	};

	//! D_4 Group Elements.
	enum class D4 : unsigned char { Id, Rot90, Rot180, Rot270, FlipX, FlipY, Diag, AntiDiag };
	//! D_4 symmetry transformation on Coordinate.
	const auto mapCoord = [boardSize](Coord c, D4 g) -> Coord {
		switch (g) {
		case D4::Id:
			return c;
		case D4::Rot90:
			return {boardSize - 1u - c.y, c.x};
		case D4::Rot180:
			return {boardSize - 1u - c.x, boardSize - 1u - c.y};
		case D4::Rot270:
			return {c.y, boardSize - 1u - c.x};
		case D4::FlipX:
			return {boardSize - 1u - c.x, c.y};
		case D4::FlipY:
			return {c.x, boardSize - 1u - c.y};
		case D4::Diag:
			return {c.y, c.x};
		case D4::AntiDiag:
			return {boardSize - 1u - c.y, boardSize - 1u - c.x};
		}
		return c;
	};

	constexpr std::array<D4, 8> group = {
	        D4::Id, D4::Rot90, D4::Rot180, D4::Rot270, D4::FlipX, D4::FlipY, D4::Diag, D4::AntiDiag,
	};


	// Find proper symmetry transformation.
	const auto symmetryIt = std::find_if(group.begin(), group.end(), [&](D4 g) {
		const Coord mapped = mapCoord(placedCoord, g);
		return mapped.x == gaugeCoord.x && mapped.y == gaugeCoord.y;
	});
	if (symmetryIt == group.end()) {
		return false;
	}

	// Transform the geometry:
	//  - Construct trafo matrix Tg = T[g] (representation of the group element g)
	//  - Transform homography   H' = T[g] H
	//  - Transform image      i_B' = T[g] i_B = (T[g] H) i
	//  - Transform intersect. p'_a = T[g] p_a
	// A_g \equiv T[g]
	const D4 g = *symmetryIt;
	if (g != D4::Id) {
		const int width  = geometry.imageB.cols;
		const int height = geometry.imageB.rows;
		cv::Mat Tg       = cv::Mat::eye(3, 3, CV_64F); //!< Representation of g: T[g]  -> Transformation matrix for H
		cv::Mat imageTransformed;                      //!< Transformed with a representation of our group action i_B' = T[g] i_B = (T[g] H) i

		switch (g) {
		case D4::Id:
			break;
		case D4::Rot90:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_90_CLOCKWISE);
			Tg = (cv::Mat_<double>(3, 3) << 0.0, -1.0, static_cast<double>(height - 1), 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::Rot180:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_180);
			Tg = (cv::Mat_<double>(3, 3) << -1.0, 0.0, static_cast<double>(width - 1), 0.0, -1.0, static_cast<double>(height - 1), 0.0, 0.0, 1.0);
			break;
		case D4::Rot270:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_90_COUNTERCLOCKWISE);
			Tg = (cv::Mat_<double>(3, 3) << 0.0, 1.0, 0.0, -1.0, 0.0, static_cast<double>(width - 1), 0.0, 0.0, 1.0);
			break;
		case D4::FlipX:
			cv::flip(geometry.imageB, imageTransformed, 1);
			Tg = (cv::Mat_<double>(3, 3) << -1.0, 0.0, static_cast<double>(width - 1), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::FlipY:
			cv::flip(geometry.imageB, imageTransformed, 0);
			Tg = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, static_cast<double>(height - 1), 0.0, 0.0, 1.0);
			break;
		case D4::Diag:
			cv::transpose(geometry.imageB, imageTransformed);
			Tg = (cv::Mat_<double>(3, 3) << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::AntiDiag:
			cv::transpose(geometry.imageB, imageTransformed);
			cv::flip(imageTransformed, imageTransformed, -1);
			Tg = (cv::Mat_<double>(3, 3) << 0.0, -1.0, static_cast<double>(height - 1), -1.0, 0.0, static_cast<double>(width - 1), 0.0, 0.0, 1.0);
			break;
		}

		// Transform the intersections
		std::vector<cv::Point2f> intersectionsTransformed;
		cv::perspectiveTransform(geometry.intersections, intersectionsTransformed, Tg);

		// Re-Order intersections coordinate (g \rhd c)
		std::vector<cv::Point2f> intersectionsOrdered(intersectionsTransformed.size());
		for (unsigned x = 0; x < boardSize; ++x) {
			for (unsigned y = 0; y < boardSize; ++y) {
				const std::size_t oldIndex     = static_cast<std::size_t>(x) * boardSize + y;
				const Coord mapped             = mapCoord({x, y}, g);
				const std::size_t newIndex     = static_cast<std::size_t>(mapped.x) * boardSize + mapped.y;
				intersectionsOrdered[newIndex] = intersectionsTransformed[oldIndex];
			}
		}

		if (geometry.H.type() != CV_64F) {
			cv::Mat H64;
			geometry.H.convertTo(H64, CV_64F);
			geometry.H = std::move(H64);
		}
		geometry.H             = Tg * geometry.H;
		geometry.imageB        = std::move(imageTransformed);
		geometry.intersections = std::move(intersectionsOrdered);
	}

	m_geometry = std::move(geometry);
	return true;
}

void Vision::setSetupImage(std::filesystem::path setupImagePath) {
	m_setupImagePath = std::move(setupImagePath);
}

void Vision::connect(Callbacks callbacks) {
	m_callbacks = std::move(callbacks);
}

void Vision::disconnect() {
	m_callbacks = {nullptr, nullptr};
}

void Vision::run() {
	if (m_running.exchange(true)) {
		return;
	}

	m_visionThread = std::thread([this]() { boardLoop(); });
}

void Vision::stop() {
	if (!m_running.exchange(false)) {
		return;
	}

	if (m_visionThread.joinable()) {
		m_visionThread.join();
	}
}


void Vision::boardLoop() {
	while (m_running.load()) {
		assert(false); // TODO: Implement
	}
}

} // namespace tengen::vision
