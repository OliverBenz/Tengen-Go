#include "vision/vision.hpp"
#include "include/vision/vision.hpp"
#include "vision/core/boardFinder.hpp"
#include "vision/core/rectifier.hpp"
#include "vision/core/stoneFinder.hpp"
#include <algorithm>


namespace tengen::vision {

Vision::Vision(Source source) : m_source{source} {
}

Vision::~Vision() {
	disconnect();
	stop();
}


bool Vision::setup(const Coord gaugeCoord) {
	const auto stoneCount = [](std::vector<core::StoneState> input) -> std::size_t {
		return std::count_if(input.begin(), input.end(), [](core::StoneState s) { return s == core::StoneState::Black || s == core::StoneState::White; });
	};
	// TODO: Setup source input


	const core::WarpResult warped = core::warpToBoard(image);
	if (core::isValidBoard(warped)) {
		// TODO: Log
		return false;
	}

	const core::BoardGeometry geometry = core::rectifyImage(image, warped);
	if (!core::isValidGeometry(geometry)) {
		// TODO: Log
		return false;
	}

	const core::StoneResult result = core::analyseBoard(geometry);
	if (!result.success || stoneCount(result.stones) != 1) {
		// TODO: Log
		return false;
	}

	// TODO: Break gauge symm (find g\in D_4 so the gaugeCoord matches with the placed coordinate (if possible). Then g \lhd inputImage).

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
