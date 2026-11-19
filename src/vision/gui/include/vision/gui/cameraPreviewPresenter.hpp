#pragma once

#include <QCameraDevice>
#include <QList>
#include <QObject>

namespace tengen::vision::gui {

class CameraPreviewView;
class CameraSource;

//! Lets the user pick a camera and watch it. Starts with the system's default camera.
class CameraPreviewPresenter : public QObject {
	Q_OBJECT

public:
	CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera);

private:
	void refreshDevices(); //!< List the cameras again and start the default one.

private:
	CameraPreviewView& m_view;
	CameraSource& m_camera;
	QList<QCameraDevice> m_devices; //!< As shown in the view.
};

} // namespace tengen::vision::gui
