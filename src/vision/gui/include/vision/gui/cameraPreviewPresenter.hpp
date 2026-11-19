#pragma once

#include <QCameraDevice>
#include <QList>
#include <QObject>
#include <QPointF>

namespace tengen::vision::gui {

class CameraPreviewView;
class CameraSource;

//! Lets the user pick a camera, watch it and pick the board corners on it. Starts with the system's default camera.
class CameraPreviewPresenter : public QObject {
	Q_OBJECT

public:
	CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera);

private slots:
	void chooseDevice(int index);
	void refreshDevices(); //!< List the cameras again and start the default one.
	void pickCorner(QPointF corner);
	void redo(); //!< Drop the corners.

private:
	void showProgress();

private:
	CameraPreviewView& m_view;      //!< Shows what we tell it.
	CameraSource& m_camera;         //!< Delivers the frames.
	QList<QCameraDevice> m_devices; //!< As shown in the view.
	QList<QPointF> m_corners;       //!< In frame coordinates, first at the game's top left.
};

} // namespace tengen::vision::gui
