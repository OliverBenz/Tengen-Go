#pragma once

#include <QImage>
#include <QPoint>
#include <QPointF>
#include <QTransform>
#include <QWidget>

#include <optional>
#include <vector>

namespace tengen::labeler {

//! Shows a photo with the corners picked so far, and a magnified view of the pixels under the mouse.
//! The mouse wheel zooms in around the mouse, dragging with the right or middle button pans.
class PhotoCanvas final : public QWidget {
	Q_OBJECT

public:
	explicit PhotoCanvas(QWidget* parent = nullptr);

	void setPhoto(const QImage& photo);
	void setCorners(const std::vector<QPoint>& boardCorners, const std::vector<QPoint>& gridCorners); //!< In photo pixels.
	void setBoardSize(unsigned size);                                                                 //!< For the grid lines drawn once all 4 grid corners are there.

signals:
	void clicked(QPoint pixel); //!< The photo pixel under a left click.

protected:
	void paintEvent(QPaintEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void wheelEvent(QWheelEvent* event) override;
	void leaveEvent(QEvent* event) override;

private:
	double scale() const; //!< Widget px per photo pixel.
	QTransform photoToWidget() const;
	std::optional<QPoint> pixelAt(QPointF widgetPoint) const; //!< None outside the photo.
	void drawCorners(QPainter& painter, const QTransform& photoToTarget) const;
	void drawLoupe(QPainter& painter, QPoint pixel) const;

private:
	QImage m_photo{};
	QImage m_scaled{};                  //!< m_photo shrunk to scale(), so moving the mouse does not shrink it on every repaint.
	double m_zoom{1.0};                 //!< On top of fitting the whole photo into the widget.
	QPointF m_centre{};                 //!< The photo point shown in the middle of the widget.
	std::optional<QPointF> m_panFrom{}; //!< Where the mouse was while dragging to pan.
	std::vector<QPoint> m_boardCorners{};
	std::vector<QPoint> m_gridCorners{};
	unsigned m_boardSize{19};
	std::optional<QPoint> m_hovered{}; //!< The photo pixel under the mouse.
};

} // namespace tengen::labeler
