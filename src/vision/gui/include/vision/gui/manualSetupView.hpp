#pragma once

#include <QImage>
#include <QPointF>
#include <QString>
#include <QWidget>

class QLabel;

namespace tengen::vision::gui {

class ImageView;

//! Lets the user pick the board corners on the camera image and shows what the pipeline makes of them.
class ManualSetupView final : public QWidget {
	Q_OBJECT

public:
	explicit ManualSetupView(QWidget* parent = nullptr);

	void showImage(const QImage& image);
	void addCorner(QPointF corner); //!< In image coordinates.
	void clearCorners();
	void showRectified(const QImage& rectified); //!< A null image clears the side column.
	void showStatus(const QString& text);

signals:
	void cornerPicked(QPointF imagePoint);
	void applyRequested();
	void redoRequested();

private:
	ImageView* m_image{nullptr};
	ImageView* m_rectified{nullptr};
	QLabel* m_status{nullptr};
};

} // namespace tengen::vision::gui
