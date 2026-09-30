#include "boardRenderer.hpp"

#include "gui/resources.hpp"

#include <QPainter>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <format>

namespace tengen::gui {

static constexpr int LINE_WIDTH = 2; //!< Grid line width [px].

BoardRenderer::BoardRenderer(const unsigned nodes)
    : m_nodes(nodes) {
	m_textureBlack = loadStone(Player::Black);
	m_textureWhite = loadStone(Player::White);
	m_ready        = m_nodes > 0 && !m_textureBlack.isNull() && !m_textureWhite.isNull();
}

unsigned BoardRenderer::nodes() const {
	return m_nodes;
}

void BoardRenderer::setNodes(unsigned nodes) {
	if (nodes == m_nodes) {
		return;
	}
	m_nodes = nodes;
	m_ready = m_nodes > 0 && !m_textureBlack.isNull() && !m_textureWhite.isNull();
	if (m_boardSizePxRequested > 0 && m_nodes > 0) {
		updateMetrics(m_boardSizePxRequested);
		updateStoneTextures();
		updateBackgroundTexture();
	}
}

void BoardRenderer::setBoardSizePx(const unsigned boardSizePx) {
	m_boardSizePxRequested = boardSizePx;
	if (boardSizePx == 0 || m_nodes == 0) {
		return;
	}
	updateMetrics(boardSizePx);
	updateStoneTextures();
	updateBackgroundTexture();
}

unsigned BoardRenderer::boardSizePx() const {
	return m_boardSize;
}

void BoardRenderer::setDevicePixelRatio(const qreal ratio) {
	if (ratio == m_devicePixelRatio) {
		return;
	}
	m_devicePixelRatio = ratio;
	updateStoneTextures();
	updateBackgroundTexture();
}

void BoardRenderer::setBackgroundTexture(const QString& path) {
	m_textureBackground = loadBoardTexture(path);
	updateBackgroundTexture();
}

void BoardRenderer::updateMetrics(const unsigned boardSizePx) {
	m_boardSize  = (boardSizePx / m_nodes) * m_nodes; // Ensure divisible by m_nodes
	m_stoneSize  = m_boardSize / m_nodes;
	m_drawStepPx = m_stoneSize / 2;
	m_coordStart = m_drawStepPx;
	m_coordEnd   = m_coordStart + (m_nodes - 1) * m_stoneSize; // Last line, exact even for odd stone sizes.
}

void BoardRenderer::updateStoneTextures() {
	if (!m_ready || m_stoneSize == 0) {
		return;
	}

	const int side = qRound(m_stoneSize * m_devicePixelRatio);
	const QSize targetSize{side, side};
	m_scaledBlack = m_textureBlack.scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	m_scaledWhite = m_textureWhite.scaled(targetSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
	m_scaledBlack.setDevicePixelRatio(m_devicePixelRatio);
	m_scaledWhite.setDevicePixelRatio(m_devicePixelRatio);
}

void BoardRenderer::updateBackgroundTexture() {
	if (m_textureBackground.isNull() || m_boardSize == 0) {
		m_scaledBackground = {};
		return;
	}

	const int side     = qRound(m_boardSize * m_devicePixelRatio);
	m_scaledBackground = m_textureBackground.scaled(side, side, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
	m_scaledBackground.setDevicePixelRatio(m_devicePixelRatio);
}

void BoardRenderer::draw(QPainter& painter, const Board& board, const Ghost& ghost) const {
	if (!isReady()) {
		return;
	}

	drawBackground(painter);
	if (ghost.draw) {
		painter.save();
		painter.setOpacity(0.45);
		drawStone(painter, ghost.coord.x, ghost.coord.y, ghost.colour);
		painter.restore();
	}
	drawStones(painter, board);
}

QRect BoardRenderer::stoneRect(const Coord coord) const {
	const int drawX = static_cast<int>((m_coordStart - m_drawStepPx) + coord.x * m_stoneSize);
	const int drawY = static_cast<int>((m_coordStart - m_drawStepPx) + coord.y * m_stoneSize);
	return {drawX, drawY, static_cast<int>(m_stoneSize), static_cast<int>(m_stoneSize)};
}

bool BoardRenderer::isReady() const {
	return m_ready && m_boardSize > 0 && m_stoneSize > 0 && !m_scaledBlack.isNull() && !m_scaledWhite.isNull();
}

void BoardRenderer::drawBackground(QPainter& painter) const {
	painter.save();
	painter.setRenderHint(QPainter::Antialiasing, true);
	if (m_scaledBackground.isNull()) {
		painter.fillRect(QRect{0, 0, static_cast<int>(m_boardSize), static_cast<int>(m_boardSize)}, PLAIN_BOARD_COLOUR);
	} else {
		painter.drawImage(QPoint{0, 0}, m_scaledBackground);
	}

	painter.setPen(QPen(Qt::black, LINE_WIDTH));
	const int effBoardWidth = static_cast<int>(m_coordEnd - m_coordStart);
	const int coordStart    = static_cast<int>(m_coordStart);
	const int coordEnd      = coordStart + effBoardWidth;
	for (unsigned i = 0; i != m_nodes; ++i) {
		const int offset = static_cast<int>(m_coordStart + i * m_stoneSize);
		painter.drawLine(coordStart, offset, coordEnd, offset);
		painter.drawLine(offset, coordStart, offset, coordEnd);
	}
	drawStarPoints(painter);
	painter.restore();
}

void BoardRenderer::drawStarPoints(QPainter& painter) const {
	if (m_nodes != 9u && m_nodes != 13u && m_nodes != 19u) {
		return;
	}

	const unsigned inset  = m_nodes >= 13u ? 3u : 2u;
	const unsigned center = m_nodes / 2u;
	const std::array<unsigned, 3> points{inset, center, m_nodes - 1u - inset};
	const qreal stoneRatio = m_nodes >= 13u ? 8.0 : 10.0;                          // Large cells on small boards need relatively smaller points.
	const qreal radius     = std::max(1.5 * LINE_WIDTH, m_stoneSize / stoneRatio); // Must stay wider than the lines to be visible.

	painter.setBrush(Qt::black);
	painter.setPen(Qt::NoPen);

	const auto drawPoint = [this, &painter, radius](const unsigned x, const unsigned y) {
		const auto centerX = static_cast<qreal>(m_coordStart + x * m_stoneSize);
		const auto centerY = static_cast<qreal>(m_coordStart + y * m_stoneSize);
		painter.drawEllipse(QPointF{centerX, centerY}, radius, radius);
	};

	drawPoint(points[0], points[0]);
	drawPoint(points[0], points[2]);
	drawPoint(points[2], points[0]);
	drawPoint(points[2], points[2]);
	drawPoint(points[1], points[1]);

	if (m_nodes == 19u) {
		drawPoint(points[0], points[1]);
		drawPoint(points[1], points[0]);
		drawPoint(points[1], points[2]);
		drawPoint(points[2], points[1]);
	}
}

void BoardRenderer::drawStone(QPainter& painter, unsigned x, unsigned y, const Board::Stone player) const {
	if (!isReady()) {
		return;
	}

	const int drawX = static_cast<int>((m_coordStart - m_drawStepPx) + x * m_stoneSize);
	const int drawY = static_cast<int>((m_coordStart - m_drawStepPx) + y * m_stoneSize);
	const QRect dest{drawX, drawY, static_cast<int>(m_stoneSize), static_cast<int>(m_stoneSize)};

	const auto& texture = (player == Board::Stone::Black) ? m_scaledBlack : m_scaledWhite;
	painter.drawImage(dest, texture);
}

void BoardRenderer::drawStones(QPainter& painter, const Board& board) const {
	for (unsigned i = 0; i != board.size(); ++i) {
		for (unsigned j = 0; j != board.size(); ++j) {
			if (board.get({i, j}) != Board::Stone::Empty) {
				drawStone(painter, i, j, board.get({i, j}));
			}
		}
	}
}

bool BoardRenderer::pixelToCoord(int pX, int pY, Coord& coord) const {
	unsigned x, y;
	if (pixelToCoord(pX, x) && pixelToCoord(pY, y)) {
		coord = {x, y};
		return true;
	}
	return false;
}

bool BoardRenderer::pixelToCoord(const int px, unsigned& coord) const {
	static constexpr float TOLERANCE = 0.4f; // To avoid accidental placement of stones.

	if (m_stoneSize == 0) {
		return false;
	}

	const auto coordRel =
	        static_cast<float>(px - static_cast<int>(m_coordStart)) / static_cast<float>(m_stoneSize); // Calculate board coordinate from pixel values.
	const auto coordRound = std::round(coordRel);                                                      // Round to nearest coordinate.

	// Click has to be close enough to a point and on the board.
	if (std::abs(coordRound - coordRel) > TOLERANCE || coordRound < 0 || coordRound > static_cast<float>(m_nodes) - 1) {
		return false;
	}

	coord = static_cast<unsigned>(coordRound);
	return true;
}

} // namespace tengen::gui
