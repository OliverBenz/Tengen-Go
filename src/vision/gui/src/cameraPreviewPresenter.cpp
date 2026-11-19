#include "vision/gui/cameraPreviewPresenter.hpp"

#include "vision/gui/cameraPreviewView.hpp"
#include "vision/gui/cameraSource.hpp"

#include <QMediaDevices>
#include <QStringList>

namespace tengen::vision::gui {

CameraPreviewPresenter::CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera)
    : m_view(view), m_camera(camera) {
	connect(&m_view, &CameraPreviewView::deviceChosen, this, [this](const int index) { m_camera.setDevice(m_devices[index]); });
	connect(&m_view, &CameraPreviewView::refreshRequested, this, &CameraPreviewPresenter::refreshDevices);
	connect(&m_camera, &CameraSource::frameReady, &m_view, &CameraPreviewView::showFrame);
	refreshDevices();
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
}

} // namespace tengen::vision::gui
