#include "gui/boardWidget.hpp"

#include "boardRenderer.hpp"

#include <QKeyEvent>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>

#include <algorithm>
#include <utility>

namespace tengen::gui {

BoardWidgetEvent BoardWidgetEvent::place(const Coord c) {
	return {Type::Place, c};
}

BoardWidgetEvent BoardWidgetEvent::pass() {
	return {Type::Pass, {0u, 0u}};
}

BoardWidgetEvent BoardWidgetEvent::resign() {
	return {Type::Resign, {0u, 0u}};
}


BoardWidget::BoardWidget(QWidget* parent)
    : QWidget(parent), m_board(9u), m_boardRenderer{std::make_unique<BoardRenderer>(static_cast<unsigned>(m_board.size()))} {
	setFocusPolicy(Qt::StrongFocus); // Required to get key events.
	setMouseTracking(true);
}

BoardWidget::~BoardWidget() = default;

const Board& BoardWidget::board() const {
	return m_board;
}

void BoardWidget::setBoard(const Board& board) {
	const auto oldSize = m_board.size();
	m_board            = board;
	if (m_board.size() != oldSize) {
		m_boardRenderer->setNodes(static_cast<unsigned>(m_board.size()));
	}
	m_boardRenderer->setBoardSizePx(boardPixelSize());
	update();
}

void BoardWidget::setCurrentPlayer(const Player player) {
	const auto currentPlayer = toStone(player);
	if (m_currentPlayer == currentPlayer) {
		return;
	}

	m_currentPlayer = currentPlayer;
	if (m_ghostStoneDraw) {
		update(stoneRect(m_ghostStone));
	}
}

const QString& BoardWidget::backgroundTexture() const {
	return m_backgroundTexture;
}

void BoardWidget::setBackgroundTexture(const QString& path) {
	m_backgroundTexture = path;
	m_boardRenderer->setBackgroundTexture(path);
	update();
}

void BoardWidget::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event);

	m_boardRenderer->setBoardSizePx(boardPixelSize());
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

void BoardWidget::mouseMoveEvent(QMouseEvent* event) {
	const auto sizePx = m_boardRenderer->boardSizePx();
	const auto local  = event->pos() - boardOffset();

	Coord newGhost{};
	// Ghost is valid when 1) mouse in board area 2) Mouse maps to a valid coordinate.
	bool ghostValid = sizePx != 0u && local.x() >= 0 && local.y() >= 0 && local.x() < static_cast<int>(sizePx) && local.y() < static_cast<int>(sizePx);
	if (ghostValid) {
		ghostValid &= m_boardRenderer->pixelToCoord(local.x(), local.y(), newGhost);
	}

	// Update only if old state differs from new
	if (m_ghostStoneDraw != ghostValid || (ghostValid && (m_ghostStone.x != newGhost.x || m_ghostStone.y != newGhost.y))) {
		const auto oldGhost     = m_ghostStone;
		const bool oldGhostDraw = m_ghostStoneDraw;
		m_ghostStone            = newGhost;
		m_ghostStoneDraw        = ghostValid;
		if (oldGhostDraw) {
			update(stoneRect(oldGhost));
		}
		if (m_ghostStoneDraw) {
			update(stoneRect(m_ghostStone));
		}
	}
	event->accept();
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
	const auto sizePx = m_boardRenderer->boardSizePx();
	const auto local  = pos - boardOffset();
	if (sizePx == 0u) {
		return;
	}

	// Clicked in bounds
	if (local.x() < 0 || local.y() < 0 || local.x() >= static_cast<int>(sizePx) || local.y() >= static_cast<int>(sizePx)) {
		return;
	}

	// Try push event
	Coord coord{};
	if (m_boardRenderer->pixelToCoord(local.x(), local.y(), coord)) {
		emit boardEvent(BoardWidgetEvent::place(coord));
	}
}

QRect BoardWidget::stoneRect(const Coord coord) const {
	return m_boardRenderer->stoneRect(coord).translated(boardOffset());
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

	const auto boardSize = static_cast<unsigned>(m_board.size());
	if (m_boardRenderer->nodes() != boardSize) {
		m_boardRenderer->setNodes(boardSize);
		m_boardRenderer->setBoardSizePx(size);
	}
	// The board lies on the window like an object: window colour around it and a soft shadow below.
	const auto boardPx = static_cast<int>(m_boardRenderer->boardSizePx());
	const QRect boardRect{boardOffset(), QSize{boardPx, boardPx}};

	QPainter painter(this);
	painter.fillRect(rect(), palette().window());
	drawShadow(painter, boardRect);

	painter.save();
	painter.translate(boardRect.topLeft());
	m_boardRenderer->draw(painter, m_board, {m_ghostStone, m_currentPlayer, m_ghostStoneDraw});
	painter.restore();

	painter.setPen(QColor(0, 0, 0, 60)); // Thin edge so light boards stand out on light themes.
	painter.drawRect(boardRect.adjusted(0, 0, -1, -1));
}

void BoardWidget::drawShadow(QPainter& painter, const QRect& boardRect) const {
	static constexpr int LAYERS = 10;     //!< Blur width [px]. Each layer adds a little darkness.
	static constexpr QPoint OFFSET{2, 4}; //!< Light comes from the top left.
	static const QColor layerColour{0, 0, 0, 7};

	painter.save();
	painter.setRenderHint(QPainter::Antialiasing, true);
	painter.setPen(Qt::NoPen);
	painter.setBrush(layerColour);
	for (int i = LAYERS; i > 0; --i) {
		painter.drawRoundedRect(QRectF(boardRect.translated(OFFSET)).adjusted(-i, -i, i, i), i, i);
	}
	painter.restore();
}

unsigned BoardWidget::boardPixelSize() const {
	const auto side = std::min(width(), height()) - 2 * BOARD_MARGIN;
	return static_cast<unsigned>(std::max(side, 0));
}

QPoint BoardWidget::boardOffset() const {
	const auto boardSize = static_cast<int>(m_boardRenderer->boardSizePx());
	const int dx         = (width() - boardSize) / 2;
	const int dy         = (height() - boardSize) / 2;
	return {dx, dy};
}

} // namespace tengen::gui
