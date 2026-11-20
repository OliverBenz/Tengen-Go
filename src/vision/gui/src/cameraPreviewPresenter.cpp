#include "vision/gui/cameraPreviewPresenter.hpp"

#include "vision/gui/cameraPreviewView.hpp"
#include "vision/gui/cameraSource.hpp"

#include <QMediaDevices>
#include <QStringList>

namespace tengen::vision::gui {

static Image toImage(const QImage& image) {
	const QImage rgb = image.convertToFormat(QImage::Format_RGB888);
	Image result{.width = rgb.width(), .height = rgb.height(), .rgb = {}};
	for (int y = 0; y < rgb.height(); ++y) { // Its rows may be padded, an Image's are not.
		result.rgb.insert(result.rgb.end(), rgb.constScanLine(y), rgb.constScanLine(y) + qsizetype{3} * rgb.width());
	}
	return result;
}

//! A copy that owns its pixels, so it outlives the image.
static QImage toQImage(const Image& image) {
	return QImage(image.rgb.data(), image.width, image.height, qsizetype{3} * image.width, QImage::Format_RGB888).copy();
}

CameraPreviewPresenter::CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera)
    : m_view(view), m_camera(camera) {
	connect(&m_view, &CameraPreviewView::deviceChosen, this, &CameraPreviewPresenter::chooseDevice);
	connect(&m_view, &CameraPreviewView::refreshRequested, this, &CameraPreviewPresenter::refreshDevices);
	connect(&m_view, &CameraPreviewView::cornerPicked, this, &CameraPreviewPresenter::pickCorner);
	connect(&m_view, &CameraPreviewView::redoRequested, this, &CameraPreviewPresenter::redo);
	connect(&m_camera, &CameraSource::frameReady, this, &CameraPreviewPresenter::showFrame);
#ifndef NDEBUG
	m_testFrame = QImage(QStringLiteral(TENGEN_TEST_IMAGE));
	m_testTimer.setInterval(100);
	connect(&m_testTimer, &QTimer::timeout, this, [this] { showFrame(m_testFrame); });
#endif
	m_device = QMediaDevices::defaultVideoInput();
	m_camera.setDevice(m_device);
#ifndef NDEBUG
	if (m_device.isNull() && !m_testFrame.isNull()) { // No camera: start on the test image.
		m_testTimer.start();
	}
#endif
	refreshDevices();
	showProgress();
}

void CameraPreviewPresenter::chooseDevice(const int index) {
#ifndef NDEBUG
	if (index == m_devices.size()) { // The test image.
		if (!m_testTimer.isActive()) {
			m_device = {};
			m_camera.setDevice(m_device);
			m_testTimer.start();
			redo();
		}
		return;
	}
	if (m_testTimer.isActive()) {
		m_testTimer.stop();
		m_device = {}; // So the chosen camera starts below.
	}
#endif
	if (m_devices[index] == m_device) {
		return;
	}
	m_device = m_devices[index];
	m_camera.setDevice(m_device);
	redo(); // Picked on the other camera's image.
}

void CameraPreviewPresenter::refreshDevices() {
	m_devices = QMediaDevices::videoInputs();
	QStringList names;
	for (const QCameraDevice& device: m_devices) {
		names.append(device.description());
	}
	int selected = static_cast<int>(m_devices.indexOf(m_device));
#ifndef NDEBUG
	if (!m_testFrame.isNull()) {
		names.append(tr("Test image"));
		if (m_testTimer.isActive()) {
			selected = static_cast<int>(m_devices.size());
		}
	}
#endif
	m_view.showDevices(names, selected);
}

void CameraPreviewPresenter::pickCorner(const QPointF corner) {
	if (m_picked < CORNER_COUNT) {
		m_corners[m_picked++] = Point{static_cast<float>(corner.x()), static_cast<float>(corner.y())};
		m_view.addCorner(corner);
		if (m_picked == CORNER_COUNT) {
			m_view.clearCorners(); // They mark the frame, not the warped board.
		}
		showProgress();
	}
}

void CameraPreviewPresenter::redo() {
	m_picked = 0;
	m_view.clearCorners();
	showProgress();
}

void CameraPreviewPresenter::showFrame(const QImage& frame) {
	if (frame.isNull()) { // E.g. when the camera stops.
		return;
	}
	if (m_picked < CORNER_COUNT) {
		m_view.showFrame(frame);
		return;
	}
	m_view.showFrame(toQImage(warpToBoard(toImage(frame), m_corners)));
}

void CameraPreviewPresenter::showProgress() {
	if (m_picked == CORNER_COUNT) {
		m_view.showStatus(tr("Board selected. Redo to pick other corners."));
	} else {
		m_view.showStatus(tr("Click the %1 corners of the board, first the one the game shows at the top left: %2 picked.").arg(CORNER_COUNT).arg(m_picked));
	}
}

} // namespace tengen::vision::gui
