#pragma once

#include "gui/resources.hpp"
#include "gui/soundPlayer.hpp"
#include "model/board.hpp"

#include <QWidget>
#include <cstdint>

#include <memory>
#include <optional>

namespace tengen::gui {

class BoardRenderer;

struct BoardWidgetEvent {
	enum class Type : std::uint8_t {
		Place,
		Pass,
		Resign,
	};

	Type type{Type::Place};
	Coord coord{0u, 0u};

	static BoardWidgetEvent place(const Coord c);
	static BoardWidgetEvent pass();
	static BoardWidgetEvent resign();
};

class BoardWidget : public QWidget {
	Q_OBJECT

public:
	static constexpr int BOARD_MARGIN = 14; //!< Space around the board for its shadow [px].

	explicit BoardWidget(const Style& initialStyle, QWidget* parent = nullptr);
	~BoardWidget();

	const Board& board() const;
	void setBoard(const Board& board);
	void setCurrentPlayer(Player player);

	void playStonePlaceSound();

signals:
	void boardEvent(const BoardWidgetEvent& event);

public slots:
	void applyStyle(const Style& style); //!< Apply every style value to the renderer and sound player.

protected:
	void resizeEvent(QResizeEvent* event) override;
	void paintEvent(QPaintEvent* event) override;
	void mouseMoveEvent(QMouseEvent* event) override;
	void mouseReleaseEvent(QMouseEvent* event) override;
	void keyReleaseEvent(QKeyEvent* event) override;

private:
	void handleClick(const QPoint& pos); //!< Resolve click position to board coordinate and emit an event if valid.
	QRect stoneRect(Coord coord) const;  //!< Get the rectangle around a stone at given coordinates.
	void renderBoard();
	void drawShadow(QPainter& painter, const QRect& boardRect) const; //!< Soft shadow below the board.

	unsigned boardPixelSize() const; //!< Space available for the board in pixels.
	QPoint boardOffset() const;      //!< Offset of the drawn board's top left corner that centers it in the widget.

private:
	Board m_board;
	Board::Stone m_currentPlayer{Board::Stone::Black};
	std::unique_ptr<BoardRenderer> m_boardRenderer;
	SoundPlayer m_soundPlayer;
	std::optional<Style> m_appliedStyle; //!< Previous style to only reload the fields that actually changed. (optional to handle first load)

	// Ghost stone: A translucent stone on mouse position to show where the placement is done.
	Coord m_ghostStone{0u, 0u};
	bool m_ghostStoneDraw = false;
};

} // namespace tengen::gui
