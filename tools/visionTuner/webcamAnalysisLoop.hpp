#pragma once

#include "analyser.hpp"
#include "mainWindow.hpp"

#include <QTimer>

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>

namespace tengen {

//! Captures webcam frames periodically, runs analysis, and updates the UI.
class WebcamAnalysisLoop {
public:
	WebcamAnalysisLoop(MainWindow& window, vision::Analyser& analyser, int cameraIndex = 0, int periodMs = 500);
	~WebcamAnalysisLoop();

	bool start();
	void stop();

	//! Re-render using the last captured frame (used when pipeline step changes).
	void refreshFromLastFrame();

private:
	void captureAndDisplay();
	void displayAnalysed(const cv::Mat& frame);

private:
	MainWindow& m_window;
	vision::Analyser& m_analyser;
	cv::VideoCapture m_capture{};
	cv::Mat m_lastFrame{};
	QTimer m_timer{};
	int m_cameraIndex{0};
};

} // namespace tengen
