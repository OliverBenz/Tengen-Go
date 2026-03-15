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

signals:
	void boardEvent(const BoardWidgetEvent& event);

protected:
	void resizeEvent(QResizeEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

private:
	//! Resolve click position to board coordinate and emit an event if valid.
	void handleClick(const QPoint& pos);
	void renderBoard();

	//! Get the board size in pixels.
	unsigned boardPixelSize() const;
	//! Offset to get to the center of the board for drawing.
	QPoint boardOffset(unsigned boardSize) const;

private:
	Board m_board;
	std::unique_ptr<BoardRenderer> m_boardRenderer;
};

} // namespace tengen::gui
