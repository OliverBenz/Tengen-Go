#pragma once

#include "model/board.hpp"

#include <QWidget>

#include <memory>

namespace tengen::gui {

class BoardRenderer;

struct BoardWidgetEvent {
	enum class Type { Place, Pass, Resign };

	Type type{Type::Place};
	Coord coord{0u, 0u};

	static BoardWidgetEvent place(const Coord c);
	static BoardWidgetEvent pass();
	static BoardWidgetEvent resign();
};

class BoardWidget : public QWidget {
	Q_OBJECT

public:
	explicit BoardWidget(QWidget* parent = nullptr);
	~BoardWidget();

	const Board& board() const;
	void setBoard(const Board& board);
	void setCurrentPlayer(Player player);

signals:
	void boardEvent(const BoardWidgetEvent& event);

protected:
	void resizeEvent(QResizeEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

private:
	//! Resolve click position to board coordinate and emit an event if valid.
	void handleClick(const QPoint& pos);
	QRect stoneRect(Coord coord) const; //!< Get the rectangle around a stone at given coordinates.
	void renderBoard();

	//! Get the board size in pixels.
	unsigned boardPixelSize() const;
	//! Offset to get to the center of the board for drawing.
	QPoint boardOffset(unsigned boardSize) const;

private:
	Board m_board;
	Board::Stone m_currentPlayer{Board::Stone::Black};
	std::unique_ptr<BoardRenderer> m_boardRenderer;

	// Ghost stone: A translucent stone on mouse position to show where the placement is done.
	Coord m_ghostStone{0u, 0u};
	bool m_ghostStoneDraw = false;
};

} // namespace tengen::gui
