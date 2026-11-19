#pragma once

#include <QCamera>
#include <QCameraDevice>
#include <QImage>
#include <QMediaCaptureSession>
#include <QObject>
#include <QVideoSink>

namespace tengen::vision::gui {

//! The camera as a stream of frames. Off until a device is set.
class CameraSource : public QObject {
	Q_OBJECT

public:
	explicit CameraSource(QObject* parent = nullptr);

	void setDevice(const QCameraDevice& device); //!< A null device turns the camera off.

signals:
	void frameReady(const QImage& frame);

private:
	QCamera m_camera;
	QVideoSink m_sink;
	QMediaCaptureSession m_session; //!< Last: lets go of the camera and sink first.
};

} // namespace tengen::vision::gui
