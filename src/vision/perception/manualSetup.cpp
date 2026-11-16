#include "vision/manualSetup.hpp"

#include "camera.hpp"
#include "stoneDebounce.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/gridFinder.hpp"
#include "vision/core/stoneFinder.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <mutex>
#include <optional>
#include <utility>

namespace tengen::vision {

static constexpr std::chrono::milliseconds FRAME_PERIOD{100}; //!< Short enough for the debounce to settle within a second.

static Image toImage(const cv::Mat& bgr) {
	if (bgr.empty()) {
		return {};
	}

	cv::Mat rgb;
	cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
	return {.width = rgb.cols, .height = rgb.rows, .rgb = std::vector<std::uint8_t>(rgb.data, rgb.data + rgb.total() * rgb.elemSize())};
}

//! The rectified board with its grid points dotted green and every stone circled: red for black, blue for white.
static cv::Mat markDetections(const core::RectifiedBoard& board, const std::vector<core::StoneState>& stones) {
	const core::BoardGeometry& geometry = board.geometry;

	cv::Mat marked = board.imageB.clone();
	for (const cv::Point2f& point: geometry.intersections) {
		cv::circle(marked, point, std::max(2, static_cast<int>(geometry.spacing * 0.08)), cv::Scalar(0, 200, 0), cv::FILLED, cv::LINE_AA);
	}
	for (std::size_t i = 0; i < stones.size(); ++i) {
		if (stones[i] != core::StoneState::Empty) {
			const cv::Scalar colour = stones[i] == core::StoneState::Black ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
			cv::circle(marked, geometry.intersections[i], static_cast<int>(geometry.spacing * 0.45), colour, 2, cv::LINE_AA);
		}
	}
	return marked;
}

struct ManualSetup::State {
	State(int cameraIndex, const std::filesystem::path& modelPath, Callbacks callbacks);
	void onFrame(const cv::Mat& frame);                                                             //!< Camera thread.
	BoardSnapshot analyse(const cv::Mat& frame, const std::array<cv::Point2f, CORNER_COUNT>& quad); //!< Camera thread.

	Callbacks report;
	std::mutex mutex;
	std::array<cv::Point2f, CORNER_COUNT> corners{};
	std::size_t picked{0}; //!< Corners picked so far.
	bool applied{false};
	cv::dnn::Net model;                      //!< Camera thread only.
	std::optional<core::BoardGeometry> grid; //!< Camera thread only. The grid within the applied corners, once found.
	StoneDebounce debounce;                  //!< Camera thread only.
	Camera camera;                           //!< Last: starts after and stops before the members it reports to.
};

ManualSetup::State::State(const int cameraIndex, const std::filesystem::path& modelPath, Callbacks callbacks)
    : report(std::move(callbacks)), model(core::loadStoneModel(modelPath)), camera(cameraIndex, FRAME_PERIOD, {.onFrame = [this](const cv::Mat& frame) { onFrame(frame); }, .onLost = [this] { report.onCameraLost(); }}) {
	assert(report.onFrame && report.onAnalysed && report.onCameraLost);
}

void ManualSetup::State::onFrame(const cv::Mat& frame) {
	bool analysing = false;
	std::array<cv::Point2f, CORNER_COUNT> quad{};
	{
		std::lock_guard<std::mutex> lock(mutex);
		analysing = applied;
		quad      = corners;
	}

	// Analysing on the camera thread is fine: the camera drains its buffer again until the next period is due.
	if (analysing) {
		report.onAnalysed(analyse(frame, quad));
	} else {
		// The next corners may show another board.
		grid.reset();
		debounce.reset();
		report.onFrame(toImage(frame));
	}
}

// Runs the pipeline from H_0 on: warp, find the grid, find the stones and steady them over the last frames.
// The grid is only looked for until one is found: the board stays where it is for a game (see vision/core/README.md),
// while a hand placing a stone would hide the grid or bend it.
BoardSnapshot ManualSetup::State::analyse(const cv::Mat& frame, const std::array<cv::Point2f, CORNER_COUNT>& quad) {
	const core::WarpResult warped = core::warpToCorners(frame, quad);
	if (!grid) {
		core::BoardGeometry found = core::analyseGeometry(warped);
		if (!core::isValidGeometry(found)) {
			return {.warped = toImage(warped.imageB0), .rectified = {}, .boardSize = 0u, .blackStones = 0u, .whiteStones = 0u};
		}
		grid = std::move(found);
	}

	const core::RectifiedBoard board = core::transformImage(frame, *grid);
	const auto& stones               = debounce.update(core::analyseBoard(board, model).stones);
	return {
	        .warped      = toImage(warped.imageB0),
	        .rectified   = toImage(markDetections(board, stones)),
	        .boardSize   = grid->boardSize,
	        .blackStones = static_cast<unsigned>(std::ranges::count(stones, core::StoneState::Black)),
	        .whiteStones = static_cast<unsigned>(std::ranges::count(stones, core::StoneState::White)),
	};
}


ManualSetup::ManualSetup(const int cameraIndex, const std::filesystem::path& modelPath, Callbacks callbacks)
    : m_state(std::make_unique<State>(cameraIndex, modelPath, std::move(callbacks))) {
}

ManualSetup::~ManualSetup() = default;

bool ManualSetup::pick(const float x, const float y) {
	std::lock_guard<std::mutex> lock(m_state->mutex);
	if (m_state->picked == CORNER_COUNT) {
		return false;
	}
	m_state->corners[m_state->picked++] = cv::Point2f(x, y);
	return true;
}

std::size_t ManualSetup::picked() const {
	std::lock_guard<std::mutex> lock(m_state->mutex);
	return m_state->picked;
}

bool ManualSetup::apply() {
	std::lock_guard<std::mutex> lock(m_state->mutex);
	m_state->applied = m_state->picked == CORNER_COUNT;
	return m_state->applied;
}

void ManualSetup::redo() {
	std::lock_guard<std::mutex> lock(m_state->mutex);
	m_state->picked  = 0;
	m_state->applied = false;
}

bool ManualSetup::isApplied() const {
	std::lock_guard<std::mutex> lock(m_state->mutex);
	return m_state->applied;
}

} // namespace tengen::vision
