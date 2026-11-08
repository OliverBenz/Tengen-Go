#include "imageView.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>
#include <QTransform>

namespace tengen::vision::gui {

static constexpr double MARKER_RADIUS = 5.0; //!< [px]

//! Maps image coordinates into the rectangle the image is drawn in.
static QTransform imageToWidget(const QRectF& target, const QSize imageSize) {
	const double scale = target.width() / imageSize.width();
	return QTransform::fromTranslate(target.x(), target.y()).scale(scale, scale);
}

ImageView::ImageView(QWidget* parent)
    : QWidget(parent) {
	setMinimumSize(160, 120);
}

void ImageView::setImage(const QImage& image) {
	m_image = image;
	update();
}

void ImageView::addMarker(const QPointF marker) {
	m_markers.push_back(marker);
	update();
}

void ImageView::clearMarkers() {
	m_markers.clear();
	update();
}

void ImageView::paintEvent(QPaintEvent*) {
	QPainter painter(this);
	painter.fillRect(rect(), palette().window());
	if (m_image.isNull()) {
		return;
	}

	const QRectF target = imageRect();
	painter.setRenderHint(QPainter::SmoothPixmapTransform);
	painter.drawImage(target, m_image);

	const QTransform toWidget = imageToWidget(target, m_image.size());
	painter.setRenderHint(QPainter::Antialiasing);
	painter.setPen(QPen(Qt::green, 2.0));
	for (const QPointF& marker: m_markers) {
		painter.drawEllipse(toWidget.map(marker), MARKER_RADIUS, MARKER_RADIUS);
	}
}

void ImageView::mousePressEvent(QMouseEvent* event) {
	const QRectF target = imageRect();
	if (event->button() != Qt::LeftButton || m_image.isNull() || !target.contains(event->position())) {
		return;
	}
	emit clicked(imageToWidget(target, m_image.size()).inverted().map(event->position()));
}

void ImageView::mouseDoubleClickEvent(QMouseEvent*) {
	// Its first press already counted. Qt's default would report the second one as another press.
}

QRectF ImageView::imageRect() const {
	const QSizeF fitted = QSizeF(m_image.size()).scaled(size(), Qt::KeepAspectRatio);
	return {QPointF((width() - fitted.width()) / 2.0, (height() - fitted.height()) / 2.0), fitted};
}

} // namespace tengen::vision::gui
