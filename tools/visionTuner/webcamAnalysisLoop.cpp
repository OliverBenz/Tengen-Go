#include "webcamAnalysisLoop.hpp"

#include <QObject>

namespace tengen {

WebcamAnalysisLoop::WebcamAnalysisLoop(MainWindow& window, vision::Analyser& analyser, const int cameraIndex, const int periodMs)
    : m_window(window), m_analyser(analyser), m_cameraIndex(cameraIndex) {
	m_timer.setInterval(periodMs);
	m_timer.setTimerType(Qt::PreciseTimer);
	QObject::connect(&m_timer, &QTimer::timeout, [&]() { captureAndDisplay(); });
}

WebcamAnalysisLoop::~WebcamAnalysisLoop() {
	stop();
}

bool WebcamAnalysisLoop::start() {
	if (m_capture.isOpened()) {
		return true;
	}

	if (!m_capture.open(m_cameraIndex)) {
		return false;
	}

	captureAndDisplay();
	m_timer.start();
	return true;
}

void WebcamAnalysisLoop::stop() {
	m_timer.stop();
	if (m_capture.isOpened()) {
		m_capture.release();
	}
}

void WebcamAnalysisLoop::refreshFromLastFrame() {
	if (m_lastFrame.empty()) {
		return;
	}
	displayAnalysed(m_lastFrame);
}

void WebcamAnalysisLoop::captureAndDisplay() {
	if (!m_capture.isOpened()) {
		return;
	}

	cv::Mat frame;
	if (!m_capture.read(frame) || frame.empty()) {
		return;
	}

	m_lastFrame = frame;
	displayAnalysed(frame);
}

void WebcamAnalysisLoop::displayAnalysed(const cv::Mat& frame) {
	m_window.setImage(m_analyser.analyse(frame, m_window.selectedPipelineStep()));
}

} // namespace tengen
