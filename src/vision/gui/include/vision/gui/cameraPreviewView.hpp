#pragma once

#include <QImage>
#include <QPointF>
#include <QStringList>
#include <QWidget>

class QComboBox;
class QLabel;

namespace tengen::vision::gui {

class ImageView;

//! Shows a camera image, with the choice of camera at the top right. Clicks on the image pick the board corners.
class CameraPreviewView : public QWidget {
	Q_OBJECT

public:
	explicit CameraPreviewView(QWidget* parent = nullptr);

	void showDevices(const QStringList& names, int selected); //!< No names greys the choice out and hides the rest.
	void showFrame(const QImage& frame);
	void addCorner(QPointF corner); //!< In frame coordinates.
	void clearCorners();
	void showStatus(const QString& text);

signals:
	void deviceChosen(int index);
	void refreshRequested();
	void cornerPicked(QPointF framePoint);
	void redoRequested();

private:
	QComboBox* m_devices{nullptr}; //!< Camera choice.
	QWidget* m_preview{nullptr};   //!< Image, status and redo. Hidden without a camera.
	ImageView* m_image{nullptr};   //!< Camera frame with the picked corners.
	QLabel* m_status{nullptr};     //!< Picking progress.
};

} // namespace tengen::vision::gui
