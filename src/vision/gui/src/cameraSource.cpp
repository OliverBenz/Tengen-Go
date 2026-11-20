#include "vision/gui/cameraSource.hpp"

#include <QVideoFrame>

namespace tengen::vision::gui {

CameraSource::CameraSource(QObject* parent)
    : QObject(parent) {
	m_session.setCamera(&m_camera);
	m_session.setVideoSink(&m_sink);
	connect(&m_sink, &QVideoSink::videoFrameChanged, this, [this](const QVideoFrame& frame) { emit frameReady(frame.toImage()); });
}

void CameraSource::setDevice(const QCameraDevice& device) {
	m_camera.setCameraDevice(device);
	m_camera.setActive(!device.isNull());
}

} // namespace tengen::vision::gui
