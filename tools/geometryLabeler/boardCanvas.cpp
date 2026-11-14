#include "boardCanvas.hpp"

#include "grid.hpp"

#include <QMouseEvent>
#include <QPainter>
#include <QPen>

#include <cmath>

namespace tengen::labeler {

static constexpr double MARKER_RADIUS = 0.3; //!< [spacings] Smaller than a stone, so the photographed stone stays visible.

BoardCanvas::BoardCanvas(QWidget* parent)
    : QWidget(parent) {
	setMinimumSize(320, 320);
	setCursor(Qt::CrossCursor);
}

void BoardCanvas::setBoard(const QImage& board, const unsigned size) {
	m_board = board;
	m_size  = size;
	update();
}

void BoardCanvas::setStones(const std::vector<Stone>& stones) {
	m_stones = stones;
	update();
}

void BoardCanvas::paintEvent(QPaintEvent*) {
	QPainter painter(this);
	painter.fillRect(rect(), palette().window());
	if (m_board.isNull()) {
		return;
	}

	painter.setRenderHint(QPainter::SmoothPixmapTransform);
	painter.setRenderHint(QPainter::Antialiasing);
	painter.drawImage(boardRect(), m_board);

	const QTransform toWidget = gridToWidget();
	drawGrid(painter, toWidget, m_size);
	const double radius = MARKER_RADIUS * toWidget.m11();
	for (std::size_t i = 0; i < m_stones.size(); ++i) {
		if (m_stones[i] == Stone::empty) {
			continue;
		}
		const bool black = m_stones[i] == Stone::black;
		painter.setBrush(black ? Qt::black : Qt::white);
		painter.setPen(QPen(black ? Qt::white : Qt::black, 2.0));
		const QPointF point(static_cast<double>(i % m_size), static_cast<double>(i / m_size));
		painter.drawEllipse(toWidget.map(point), radius, radius);
	}
}

void BoardCanvas::mousePressEvent(QMouseEvent* event) {
	if (event->button() != Qt::LeftButton || m_board.isNull()) {
		return;
	}
	const QPointF point = gridToWidget().inverted().map(event->position());
	const auto col      = static_cast<int>(std::lround(point.x()));
	const auto row      = static_cast<int>(std::lround(point.y()));
	if (col >= 0 && row >= 0 && col < static_cast<int>(m_size) && row < static_cast<int>(m_size)) {
		emit clicked(row, col);
	}
}

void BoardCanvas::mouseDoubleClickEvent(QMouseEvent*) {
	// Its first press already counted. Qt's default would report the second one as another press.
}

QRect BoardCanvas::boardRect() const {
	const QSize fitted = m_board.size().scaled(size(), Qt::KeepAspectRatio);
	return {QPoint((width() - fitted.width()) / 2, (height() - fitted.height()) / 2), fitted};
}

QTransform BoardCanvas::gridToWidget() const {
	const QRect target   = boardRect();
	const double spacing = target.width() / static_cast<double>(m_size + 1);
	return QTransform::fromTranslate(target.x(), target.y()).scale(spacing, spacing).translate(1.0, 1.0);
}

} // namespace tengen::labeler
