#pragma once

#include <opencv2/core/mat.hpp>

#include <chrono>
#include <functional>
#include <stop_token>
#include <thread>

namespace tengen::vision {

//! Captures camera frames on a thread of its own and hands out one per period. Destroying it waits for that thread.
class Camera {
public:
	struct Callbacks {
		std::function<void(const cv::Mat&)> onFrame; //!< A fresh BGR frame, once per period.
		std::function<void()> onLost;                //!< The camera could not be opened, stopped delivering or delivers no colour. Capturing ends.
	};

	//! Opens the camera and starts capturing. Never blocks: opening happens on the capture thread too.
	//! \note Callbacks run on the capture thread.
	Camera(int deviceIndex, std::chrono::milliseconds period, Callbacks callbacks);

private:
	void run(const std::stop_token& stopToken, int deviceIndex);

private:
	std::chrono::milliseconds m_period;
	Callbacks m_callbacks;
	std::jthread m_thread; //!< Last: starts after and stops before the members it uses.
};

} // namespace tengen::vision
