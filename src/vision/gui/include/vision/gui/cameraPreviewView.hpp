#pragma once

#include <QImage>
#include <QStringList>
#include <QWidget>

class QComboBox;
class QLabel;

namespace tengen::vision::gui {

//! Shows a camera image, with the choice of camera at the top right.
class CameraPreviewView : public QWidget {
	Q_OBJECT

public:
	explicit CameraPreviewView(QWidget* parent = nullptr);

	void showDevices(const QStringList& names, int selected); //!< No names greys the choice out and hides the image.
	void showFrame(const QImage& frame);

signals:
	void deviceChosen(int index);
	void refreshRequested();

private:
	QComboBox* m_devices{nullptr};
	QLabel* m_image{nullptr};
};

} // namespace tengen::vision::gui
