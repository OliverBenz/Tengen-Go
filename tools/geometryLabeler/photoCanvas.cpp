#include "photoCanvas.hpp"

#include "grid.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QPolygonF>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

namespace tengen::labeler {

static constexpr double MARKER_RADIUS = 4.0;  //!< [px]
static constexpr int LOUPE_PIXELS     = 41;   //!< Photo pixels per side. Odd, so the hovered pixel is the middle one.
static constexpr int LOUPE_ZOOM       = 6;    //!< Widget px per photo pixel.
static constexpr int LOUPE_MARGIN     = 10;   //!< [px] to the widget border.
static constexpr double ZOOM_STEP     = 1.25; //!< Per notch of the mouse wheel.
static constexpr double MAX_SCALE     = 16.0; //!< Widget px per photo pixel.
static constexpr QColor BOARD_COLOUR{255, 140, 0};
static constexpr QColor GRID_COLOUR{0, 220, 255};

//! Qt draws pixel x over [x, x + 1), the vision code puts its centre at x.
static QPointF pixelCentre(const QPoint pixel) {
	return QPointF(pixel) + QPointF(0.5, 0.5);
}

//! Joins the corners in click order and closes the outline once all 4 are there.
static void drawOutline(QPainter& painter, const QTransform& photoToTarget, const std::vector<QPoint>& corners, const QColor& colour) {
	QPolygonF outline;
	for (const QPoint corner: corners) {
		outline << photoToTarget.map(pixelCentre(corner));
	}
	painter.setPen(QPen(colour, 2.0));
	if (corners.size() == 4) {
		painter.drawPolygon(outline);
	} else {
		painter.drawPolyline(outline);
	}
	for (const QPointF& point: outline) {
		painter.drawEllipse(point, MARKER_RADIUS, MARKER_RADIUS);
	}
}

PhotoCanvas::PhotoCanvas(QWidget* parent)
    : QWidget(parent) {
	setMinimumSize(320, 240);
	setMouseTracking(true);
	setCursor(Qt::CrossCursor);
}

void PhotoCanvas::setPhoto(const QImage& photo) {
	m_photo  = photo;
	m_scaled = {};
	m_zoom   = 1.0;
	m_centre = QRectF(photo.rect()).center();
	m_hovered.reset();
	update();
}

void PhotoCanvas::setCorners(const std::vector<QPoint>& boardCorners, const std::vector<QPoint>& gridCorners) {
	m_boardCorners = boardCorners;
	m_gridCorners  = gridCorners;
	update();
}

void PhotoCanvas::setBoardSize(const unsigned size) {
	m_boardSize = size;
	update();
}

void PhotoCanvas::paintEvent(QPaintEvent*) {
	QPainter painter(this);
	painter.fillRect(rect(), palette().window());
	if (m_photo.isNull()) {
		return;
	}

	const QTransform toWidget = photoToWidget();
	if (scale() < 1.0) {
		const QRectF target = toWidget.mapRect(QRectF(m_photo.rect()));
		if (m_scaled.size() != target.size().toSize()) {
			m_scaled = m_photo.scaled(target.size().toSize(), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		}
		painter.drawImage(target.topLeft(), m_scaled);
	} else { // Without smoothing, so every photo pixel is a sharp square to place a corner on.
		painter.setTransform(toWidget);
		painter.drawImage(QPointF(0.0, 0.0), m_photo);
		painter.resetTransform();
	}

	painter.setRenderHint(QPainter::Antialiasing);
	drawCorners(painter, toWidget);
	if (m_hovered && scale() < LOUPE_ZOOM) {
		drawLoupe(painter, *m_hovered);
	}
}

void PhotoCanvas::mousePressEvent(QMouseEvent* event) {
	if (event->button() == Qt::RightButton || event->button() == Qt::MiddleButton) {
		m_panFrom = event->position();
		return;
	}
	if (event->button() != Qt::LeftButton) {
		return;
	}
	if (const std::optional<QPoint> pixel = pixelAt(event->position())) {
		emit clicked(*pixel);
	}
}

void PhotoCanvas::mouseDoubleClickEvent(QMouseEvent*) {
	// Its first press already counted. Qt's default would report the second one as another press.
}

void PhotoCanvas::mouseMoveEvent(QMouseEvent* event) {
	if (m_panFrom) {
		m_centre -= (event->position() - *m_panFrom) / scale();
		m_panFrom = event->position();
	}
	m_hovered = pixelAt(event->position());
	update();
}

void PhotoCanvas::mouseReleaseEvent(QMouseEvent* event) {
	if (event->button() == Qt::RightButton || event->button() == Qt::MiddleButton) {
		m_panFrom.reset();
	}
}

void PhotoCanvas::wheelEvent(QWheelEvent* event) {
	if (m_photo.isNull()) {
		return;
	}
	const QPointF mouse = event->position();
	const QPointF held  = photoToWidget().inverted().map(mouse); // Stays under the mouse while zooming.
	const double fit    = scale() / m_zoom;
	m_zoom              = std::clamp(m_zoom * std::pow(ZOOM_STEP, event->angleDelta().y() / 120.0), 1.0, std::max(1.0, MAX_SCALE / fit));
	m_centre            = m_zoom == 1.0 ? QRectF(m_photo.rect()).center() : held - (mouse - QRectF(rect()).center()) / scale();
	m_hovered           = pixelAt(mouse);
	update();
}

void PhotoCanvas::leaveEvent(QEvent*) {
	m_hovered.reset();
	update();
}

double PhotoCanvas::scale() const {
	const double fit = std::min(width() / static_cast<double>(m_photo.width()), height() / static_cast<double>(m_photo.height()));
	return fit * m_zoom;
}

QTransform PhotoCanvas::photoToWidget() const {
	const QPointF middle = QRectF(rect()).center();
	return QTransform::fromTranslate(middle.x(), middle.y()).scale(scale(), scale()).translate(-m_centre.x(), -m_centre.y());
}

std::optional<QPoint> PhotoCanvas::pixelAt(const QPointF widgetPoint) const {
	if (m_photo.isNull()) {
		return std::nullopt;
	}
	const QPointF point = photoToWidget().inverted().map(widgetPoint);
	const QPoint pixel(static_cast<int>(std::floor(point.x())), static_cast<int>(std::floor(point.y())));
	if (!m_photo.rect().contains(pixel)) {
		return std::nullopt;
	}
	return pixel;
}

void PhotoCanvas::drawCorners(QPainter& painter, const QTransform& photoToTarget) const {
	drawOutline(painter, photoToTarget, m_boardCorners, BOARD_COLOUR);
	drawOutline(painter, photoToTarget, m_gridCorners, GRID_COLOUR);
	if (m_gridCorners.size() == 4) {
		QPolygonF corners;
		for (const QPoint corner: m_gridCorners) {
			corners << pixelCentre(corner);
		}
		drawGrid(painter, gridToPhoto(corners, m_boardSize) * photoToTarget, m_boardSize);
	}
}

void PhotoCanvas::drawLoupe(QPainter& painter, const QPoint pixel) const {
	// On the half away from the mouse, so it never covers the corner being picked.
	const int size         = LOUPE_PIXELS * LOUPE_ZOOM;
	const bool mouseOnLeft = photoToWidget().map(pixelCentre(pixel)).x() < width() / 2.0;
	const QRect target(mouseOnLeft ? width() - size - LOUPE_MARGIN : LOUPE_MARGIN, LOUPE_MARGIN, size, size);
	const QRect source(pixel - QPoint(LOUPE_PIXELS / 2, LOUPE_PIXELS / 2), QSize(LOUPE_PIXELS, LOUPE_PIXELS));

	painter.save();
	painter.setClipRect(target);
	painter.fillRect(target, Qt::black);
	painter.drawImage(target, m_photo, source); // Without smoothing, so single pixels stay visible.
	drawCorners(painter, QTransform::fromTranslate(target.x(), target.y()).scale(LOUPE_ZOOM, LOUPE_ZOOM).translate(-source.x(), -source.y()));

	// A crosshair with a gap, so the pixel a click would pick stays visible.
	const QPointF centre = QRectF(target).center();
	const double gap     = LOUPE_ZOOM;
	painter.setPen(QPen(Qt::red, 1.0));
	painter.drawLine(QPointF(target.left(), centre.y()), centre - QPointF(gap, 0.0));
	painter.drawLine(centre + QPointF(gap, 0.0), QPointF(target.right() + 1, centre.y()));
	painter.drawLine(QPointF(centre.x(), target.top()), centre - QPointF(0.0, gap));
	painter.drawLine(centre + QPointF(0.0, gap), QPointF(centre.x(), target.bottom() + 1));
	painter.restore();

	painter.setPen(QPen(palette().text(), 1.0));
	painter.drawRect(target);
}

} // namespace tengen::labeler
