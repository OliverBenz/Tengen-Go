#pragma once

#include "vision/manualSetup.hpp"

#include <QCameraDevice>
#include <QImage>
#include <QList>
#include <QObject>
#include <QPointF>
#ifndef NDEBUG
#include <QTimer>
#endif

#include <array>
#include <cstddef>

namespace tengen::vision::gui {

class CameraPreviewView;
class CameraSource;

//! Lets the user pick a camera, watch it and pick the board corners on it. Shows the warped board once all are picked. Starts with the system's default camera.
class CameraPreviewPresenter : public QObject {
	Q_OBJECT

public:
	CameraPreviewPresenter(CameraPreviewView& view, CameraSource& camera);

private slots:
	void chooseDevice(int index);
	void refreshDevices(); //!< List the cameras again. Keeps the running one.
	void pickCorner(QPointF corner);
	void redo(); //!< Drop the corners.
	void showFrame(const QImage& frame);

private:
	void showProgress();

private:
	static constexpr std::size_t CORNER_COUNT = 4;

	CameraPreviewView& m_view;                   //!< Shows what we tell it.
	CameraSource& m_camera;                      //!< Delivers the frames.
	QList<QCameraDevice> m_devices;              //!< As shown in the view.
	QCameraDevice m_device;                      //!< Running, null for none.
	std::array<Point, CORNER_COUNT> m_corners{}; //!< In frame coordinates, first at the game's top left.
	std::size_t m_picked{0};                     //!< How many of them.

#ifndef NDEBUG          // To play without a camera: a still image, offered after the cameras.
	QImage m_testFrame; //!< Null if it failed to load.
	QTimer m_testTimer; //!< Repeats the image like a camera while chosen.
#endif
};

} // namespace tengen::vision::gui
