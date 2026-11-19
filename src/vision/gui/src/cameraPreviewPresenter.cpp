#include "vision/gui/cameraPreviewPresenter.hpp"

#include "vision/gui/cameraPreviewView.hpp"
#include "vision/gui/cameraSource.hpp"

#include <QMediaDevices>
#include <QStringList>

namespace tengen::vision::gui {

static constexpr qsizetype CORNER_COUNT = 4;

CameraPreviewPresenter::CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera)
    : m_view(view), m_camera(camera) {
	connect(&m_view, &CameraPreviewView::deviceChosen, this, &CameraPreviewPresenter::chooseDevice);
	connect(&m_view, &CameraPreviewView::refreshRequested, this, &CameraPreviewPresenter::refreshDevices);
	connect(&m_view, &CameraPreviewView::cornerPicked, this, &CameraPreviewPresenter::pickCorner);
	connect(&m_view, &CameraPreviewView::redoRequested, this, &CameraPreviewPresenter::redo);
	connect(&m_camera, &CameraSource::frameReady, &m_view, &CameraPreviewView::showFrame);
	refreshDevices();
}

void CameraPreviewPresenter::chooseDevice(const int index) {
	m_camera.setDevice(m_devices[index]);
	redo(); // Picked on the other camera's image.
}

void CameraPreviewPresenter::refreshDevices() {
	m_devices                     = QMediaDevices::videoInputs();
	const QCameraDevice preferred = QMediaDevices::defaultVideoInput();

	QStringList names;
	for (const QCameraDevice& device: m_devices) {
		names.append(device.description());
	}
	m_view.showDevices(names, static_cast<int>(m_devices.indexOf(preferred)));
	m_camera.setDevice(preferred);
	redo();
}

void CameraPreviewPresenter::pickCorner(const QPointF corner) {
	if (m_corners.size() < CORNER_COUNT) {
		m_corners.append(corner);
		m_view.addCorner(corner);
		showProgress();
	}
}

void CameraPreviewPresenter::redo() {
	m_corners.clear();
	m_view.clearCorners();
	showProgress();
}

void CameraPreviewPresenter::showProgress() {
	if (m_corners.size() == CORNER_COUNT) {
		m_view.showStatus(tr("Board selected. Redo to pick other corners."));
	} else {
		m_view.showStatus(tr("Click the %1 corners of the board, first the one the game shows at the top left: %2 picked.").arg(CORNER_COUNT).arg(m_corners.size()));
	}
}

} // namespace tengen::vision::gui
