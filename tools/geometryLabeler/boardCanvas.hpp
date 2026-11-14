#pragma once

#include "boardFile.hpp"

#include <QImage>
#include <QTransform>
#include <QWidget>

#include <vector>

namespace tengen::labeler {

//! Shows the board seen from above, scaled to fit, with the stones placed so far. Reports clicks as the nearest point.
class BoardCanvas final : public QWidget {
	Q_OBJECT

public:
	explicit BoardCanvas(QWidget* parent = nullptr);

	void setBoard(const QImage& board, unsigned size); //!< As made by rectify(): one spacing of margin around the grid.
	void setStones(const std::vector<Stone>& stones);  //!< Row by row.

signals:
	void clicked(int row, int col);

protected:
	void paintEvent(QPaintEvent* event) override;
	void mousePressEvent(QMouseEvent* event) override;
	void mouseDoubleClickEvent(QMouseEvent* event) override;

private:
	QRect boardRect() const;         //!< Where the board is drawn: scaled to fit, centred.
	QTransform gridToWidget() const; //!< From (col, row) of an intersection to widget px.

private:
	QImage m_board{};
	unsigned m_size{0};
	std::vector<Stone> m_stones{};
};

} // namespace tengen::labeler
