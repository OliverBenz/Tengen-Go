#pragma once

#include "gui/resources.hpp"
#include "model/board.hpp"

#include <QImage>
#include <QPainter>

namespace tengen::gui {

class BoardRenderer {
public:
	explicit BoardRenderer(unsigned nodes);

	//! Ghost stone -> Opacity stone where mouse is positioned.
	struct Ghost {
		Coord coord;         //!< Coord of ghost stone if exists.
		Board::Stone colour; //!< Colour of ghost stone.
		bool draw;           //!< Draw a ghost stone or not.
	};

	unsigned nodes() const;       //!< Board size in lines.
	unsigned boardSizePx() const; //!< Drawn board size [px]. Snapped to a multiple of the nodes.
	bool isReady() const;         //!< Textures loaded and size set.
	bool showCoordinates() const; //!< Coordinates are enabled or not.

	void setNodes(unsigned nodes);                          //!< Set the board size in lines.
	void setBoardSizePx(unsigned boardSizePx);              //!< Set the available board size [px].
	void setDevicePixelRatio(qreal ratio);                  //!< Set the display scaling. Keeps textures sharp.
	void setBackgroundTexture(boardStyle::Texture texture); //!< Set the board image.
	void setShowCoordinates(bool show);                     //!< Add a border with standard coordinates.

	void draw(QPainter& painter, const Board& board, const Ghost& ghost) const; //!< Draw board, ghost stone and stones.
	QRect stoneRect(Coord coord) const;                                         //!< Area of a stone [px].
	bool pixelToCoord(int pX, int pY, Coord& coord) const;                      //!< Convert a pixel to a board coordinate.

private:
	void updateLayout();                      //!< Apply nodes and requested size. Rescales only textures whose size changed.
	void updateMetrics(unsigned boardSizePx); //!< Compute stone size and line positions.
	void updateStoneTextures();               //!< Rescale the stones to the stone size.
	void updateBackgroundTexture();           //!< Rescale the background to the board size.

	void drawBackground(QPainter& painter) const;                                         //!< Draw background, lines and star points.
	void drawCoordinates(QPainter& painter) const;                                        //!< Draw column letters and row numbers into the board border.
	void drawStarPoints(QPainter& painter) const;                                         //!< Draw star points for standard board sizes.
	void drawStones(QPainter& painter, const Board& board) const;                         //!< Draw all stones of a board.
	void drawStone(QPainter& painter, unsigned x, unsigned y, Board::Stone player) const; //!< Draw a single stone.

	bool pixelToCoord(int px, unsigned& coord) const; //!< Convert a pixel to a coordinate on one axis.

private:
	unsigned m_nodes{0};                //!< Board size in lines.
	unsigned m_boardSizePxRequested{0}; //!< Available board size [px].
	unsigned m_boardSize{0};            //!< Drawn board size [px].
	unsigned m_stoneSize{0};            //!< Stone diameter [px].
	unsigned m_drawStepPx{0};           //!< Half a stone [px].
	unsigned m_border{0};               //!< Border between board edge and the stones for the coordinate labels [px]. Zero if hidden.
	unsigned m_coordStart{0};           //!< First line position [px].
	unsigned m_coordEnd{0};             //!< Last line position [px].
	qreal m_devicePixelRatio{1.0};      //!< Display scaling.

	QImage m_textureBlack;      //!< Black stone source.
	QImage m_textureWhite;      //!< White stone source.
	QImage m_scaledBlack;       //!< Black stone at stone size.
	QImage m_scaledWhite;       //!< White stone at stone size.
	QImage m_textureBackground; //!< Square board image source. Null draws the plain colour.
	QImage m_scaledBackground;  //!< Board image at board size.

	bool m_showCoordinates{true}; //!< Draw the coordinate border.
	bool m_ready{false};          //!< Textures loaded.
};

} // namespace tengen::gui
