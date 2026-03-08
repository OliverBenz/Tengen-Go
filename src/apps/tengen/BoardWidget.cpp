#include "BoardWidget.hpp"

#include <QColor>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>

#include <algorithm>
#include <utility>

namespace tengen::gui {

BoardWidget::BoardWidget(Board board, QWidget* parent)
    : QWidget(parent), m_board(std::move(board)), m_boardRenderer(static_cast<unsigned>(m_board.size())) {
	setFocusPolicy(Qt::StrongFocus); // Required to get key events.
	setMouseTracking(false);
}

const Board& BoardWidget::board() const {
	return m_board;
}

void BoardWidget::setBoard(const Board& board) {
	const auto oldSize = m_board.size();
	m_board            = board;
	if (m_board.size() != oldSize) {
		m_boardRenderer.setNodes(static_cast<unsigned>(m_board.size()));
	}
	m_boardRenderer.setBoardSizePx(boardPixelSize());
	update();
}

void BoardWidget::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event);

	m_boardRenderer.setBoardSizePx(boardPixelSize());
	update();
}

void BoardWidget::mouseReleaseEvent(QMouseEvent* event) {
	if (event->button() == Qt::LeftButton) {
		handleClick(event->pos());
		event->accept();
		return;
	}

	QWidget::mouseReleaseEvent(event);
}

void BoardWidget::keyReleaseEvent(QKeyEvent* event) {
	switch (event->key()) {
	case Qt::Key_P:
		emit boardEvent(BoardWidgetEvent::pass());
		event->accept();
		return;

	case Qt::Key_R:
		emit boardEvent(BoardWidgetEvent::resign());
		event->accept();
		return;

	default:
		QWidget::keyReleaseEvent(event);
		return; // Don't accept the event.
	}
}

void BoardWidget::handleClick(const QPoint& pos) {
	const auto sizePx = boardPixelSize();
	const auto local  = pos - boardOffset(sizePx);
	if (sizePx == 0u) {
		return;
	}

	// Clicked in bounds
	if (local.x() < 0 || local.y() < 0 || local.x() >= static_cast<int>(sizePx) || local.y() >= static_cast<int>(sizePx)) {
		return;
	}

	// Try push event
	Coord coord{};
	if (m_boardRenderer.pixelToCoord(local.x(), local.y(), coord)) {
		emit boardEvent(BoardWidgetEvent::place(coord));
	}
}

void BoardWidget::paintEvent(QPaintEvent* event) {
	QWidget::paintEvent(event);
	renderBoard();
}

void BoardWidget::renderBoard() {
	const auto size = boardPixelSize();
	if (size == 0u) {
		return;
	}

	const auto offset    = boardOffset(size);
	const auto boardSize = static_cast<unsigned>(m_board.size());
	if (m_boardRenderer.nodes() != boardSize) {
		m_boardRenderer.setNodes(boardSize);
		m_boardRenderer.setBoardSizePx(size);
	}
	QPainter painter(this);
	painter.fillRect(rect(), QColor(20, 20, 20));
	painter.save();
	painter.translate(offset); // Center in drawing area
	m_boardRenderer.draw(painter, m_board);
	painter.restore();
}

unsigned BoardWidget::boardPixelSize() const {
	const auto side = std::min(width(), height());
	return static_cast<unsigned>(std::max(side, 0));
}

QPoint BoardWidget::boardOffset(unsigned boardSize) const {
	const int dx = (width() - static_cast<int>(boardSize)) / 2;
	const int dy = (height() - static_cast<int>(boardSize)) / 2;
	return {dx, dy};
}

} // namespace tengen::gui
