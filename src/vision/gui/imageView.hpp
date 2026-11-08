#pragma once

#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QWidget>

#include <vector>

namespace tengen::vision::gui {

//! Shows an image scaled to fit, with markers on top. Reports clicks in image coordinates.
class ImageView final : public QWidget {
	Q_OBJECT

public:
	explicit ImageView(QWidget* parent = nullptr);

	void setImage(const QImage& image);
	void addMarker(QPointF marker); //!< In image coordinates.
	void clearMarkers();

signals:
	void clicked(QPointF imagePoint); //!< Only for left clicks on the image.

protected:
	void paintEvent(QPaintEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
	QRectF imageRect() const; //!< Where the image is drawn: scaled to fit, centred.

private:
	QImage m_image{};
	std::vector<QPointF> m_markers{};
};

} // namespace tengen::vision::gui
