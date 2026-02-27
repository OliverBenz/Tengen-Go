#include "vision/vision.hpp"
#include "include/vision/vision.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/rectifier.hpp"
#include "vision/core/stoneFinder.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>


namespace tengen::vision {

Vision::Vision(Source source) : m_source{source} {
}

Vision::~Vision() {
	disconnect();
	stop();
}


bool Vision::setup(const Coord gaugeCoord) {
	const auto stoneCount = [](const std::vector<core::StoneState>& input) -> std::size_t {
		return std::count_if(input.begin(), input.end(), [](core::StoneState s) { return s == core::StoneState::Black || s == core::StoneState::White; });
	};
	
	// Get image from source
	cv::Mat image;
	switch (m_source) {
	case Source::None:
	case Source::Image:
		return false; // Not yet implemented
	case Source::Camera: {
		cv::VideoCapture capture{0, cv::CAP_ANY};
		if (!capture.isOpened() || !capture.read(image) || image.empty()) {
			return false;
		}
		break;
	}
	}

	const core::WarpResult warped = core::warpToBoard(image);
	if (core::isValidBoard(warped)) {
		// TODO: Log
		return false;
	}

	core::BoardGeometry geometry = core::rectifyImage(image, warped);
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
	if (gaugeCoord.x >= boardSize || gaugeCoord.y >= boardSize) {
		return false;
	}

	const auto stoneIt = std::find_if(result.stones.begin(), result.stones.end(), [](core::StoneState s) {
		return s == core::StoneState::Black || s == core::StoneState::White;
	});
	if (stoneIt == result.stones.end()) {
		return false;
	}
	const std::size_t stoneIndex = static_cast<std::size_t>(std::distance(result.stones.begin(), stoneIt));
	const Coord placedCoord{
	        static_cast<unsigned>(stoneIndex / boardSize),
	        static_cast<unsigned>(stoneIndex % boardSize),
	};

	enum class D4 : unsigned char { Id, Rot90, Rot180, Rot270, FlipX, FlipY, Diag, AntiDiag };
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
	        D4::Id,
	        D4::Rot90,
	        D4::Rot180,
	        D4::Rot270,
	        D4::FlipX,
	        D4::FlipY,
	        D4::Diag,
	        D4::AntiDiag,
	};

	const auto symmetryIt = std::find_if(group.begin(), group.end(), [&](D4 g) {
		const Coord mapped = mapCoord(placedCoord, g);
		return mapped.x == gaugeCoord.x && mapped.y == gaugeCoord.y;
	});
	if (symmetryIt == group.end()) {
		return false;
	}

	const D4 symmetry = *symmetryIt;
	if (symmetry != D4::Id) {
		const int width  = geometry.imageB.cols;
		const int height = geometry.imageB.rows;
		cv::Mat imageTransformed;
		cv::Mat A = cv::Mat::eye(3, 3, CV_64F);

		switch (symmetry) {
		case D4::Id:
			break;
		case D4::Rot90:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_90_CLOCKWISE);
			A = (cv::Mat_<double>(3, 3) << 0.0, -1.0, static_cast<double>(height - 1), 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::Rot180:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_180);
			A = (cv::Mat_<double>(3, 3) << -1.0, 0.0, static_cast<double>(width - 1), 0.0, -1.0, static_cast<double>(height - 1), 0.0, 0.0,
			     1.0);
			break;
		case D4::Rot270:
			cv::rotate(geometry.imageB, imageTransformed, cv::ROTATE_90_COUNTERCLOCKWISE);
			A = (cv::Mat_<double>(3, 3) << 0.0, 1.0, 0.0, -1.0, 0.0, static_cast<double>(width - 1), 0.0, 0.0, 1.0);
			break;
		case D4::FlipX:
			cv::flip(geometry.imageB, imageTransformed, 1);
			A = (cv::Mat_<double>(3, 3) << -1.0, 0.0, static_cast<double>(width - 1), 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::FlipY:
			cv::flip(geometry.imageB, imageTransformed, 0);
			A = (cv::Mat_<double>(3, 3) << 1.0, 0.0, 0.0, 0.0, -1.0, static_cast<double>(height - 1), 0.0, 0.0, 1.0);
			break;
		case D4::Diag:
			cv::transpose(geometry.imageB, imageTransformed);
			A = (cv::Mat_<double>(3, 3) << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
			break;
		case D4::AntiDiag:
			cv::transpose(geometry.imageB, imageTransformed);
			cv::flip(imageTransformed, imageTransformed, -1);
			A = (cv::Mat_<double>(3, 3) << 0.0, -1.0, static_cast<double>(height - 1), -1.0, 0.0, static_cast<double>(width - 1), 0.0, 0.0,
			     1.0);
			break;
		}

		std::vector<cv::Point2f> intersectionsTransformed;
		cv::perspectiveTransform(geometry.intersections, intersectionsTransformed, A);

		std::vector<cv::Point2f> intersectionsOrdered(intersectionsTransformed.size());
		for (unsigned x = 0; x < boardSize; ++x) {
			for (unsigned y = 0; y < boardSize; ++y) {
				const std::size_t oldIndex = static_cast<std::size_t>(x) * boardSize + y;
				const Coord mapped         = mapCoord({x, y}, symmetry);
				const std::size_t newIndex = static_cast<std::size_t>(mapped.x) * boardSize + mapped.y;
				intersectionsOrdered[newIndex] = intersectionsTransformed[oldIndex];
			}
		}

		if (geometry.H.type() != CV_64F) {
			cv::Mat H64;
			geometry.H.convertTo(H64, CV_64F);
			geometry.H = std::move(H64);
		}
		geometry.H             = A * geometry.H;
		geometry.imageB        = std::move(imageTransformed);
		geometry.intersections = std::move(intersectionsOrdered);
	}

	m_geometry = std::move(geometry);
	return true;
}

void Vision::connect(Callbacks callback) {
	m_callbacks = std::move(callback);
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
		// TODO:
	}
}

} // namespace tengen::vision
