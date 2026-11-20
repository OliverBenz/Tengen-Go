#include "camera.hpp"

#include <opencv2/videoio.hpp>

#include <cassert>
#include <utility>

namespace tengen::vision {

Camera::Camera(const int deviceIndex, const std::chrono::milliseconds period, Callbacks callbacks)
    : m_period(period), m_callbacks(std::move(callbacks)), m_thread([this, deviceIndex](const std::stop_token& stopToken) { run(stopToken, deviceIndex); }) {
	assert(m_callbacks.onFrame && m_callbacks.onLost);
}

void Camera::run(const std::stop_token& stopToken, const int deviceIndex) {
	cv::VideoCapture capture(deviceIndex);
	auto due = std::chrono::steady_clock::now();

	while (!stopToken.stop_requested()) {
		// Grab every frame, so the driver's buffer never hands out a stale one.
		if (!capture.grab()) {
			m_callbacks.onLost();
			return;
		}
		if (std::chrono::steady_clock::now() < due) {
			continue;
		}

		cv::Mat frame;
		if (!capture.retrieve(frame) || frame.type() != CV_8UC3) {
			m_callbacks.onLost();
			return;
		}
		due = std::chrono::steady_clock::now() + m_period;
		m_callbacks.onFrame(frame);
	}
}

} // namespace tengen::vision
