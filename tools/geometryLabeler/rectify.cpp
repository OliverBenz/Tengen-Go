#include "rectify.hpp"

#include "grid.hpp"

#include <QPainter>
#include <QPointF>
#include <QPolygonF>
#include <QTransform>

namespace tengen::labeler {

QImage rectify(const QImage& photo, const Geometry& geometry, const int spacing) {
	QPolygonF corners;
	for (const Pixel corner: geometry.gridCorners) {
		corners << QPointF(corner.x + 0.5, corner.y + 0.5); // Qt draws pixel x over [x, x + 1), the label means its centre.
	}
	const QTransform gridToBoard  = QTransform::fromScale(spacing, spacing).translate(1.0, 1.0);
	const QTransform photoToBoard = gridToPhoto(corners, geometry.boardSize).inverted() * gridToBoard;

	const int side = spacing * static_cast<int>(geometry.boardSize + 1);
	QImage board(side, side, QImage::Format_RGB32);
	board.fill(Qt::black);
	QPainter painter(&board);
	painter.setRenderHint(QPainter::SmoothPixmapTransform);
	painter.setTransform(photoToBoard);
	painter.drawImage(0, 0, photo);
	painter.end();
	return board;
}

} // namespace tengen::labeler
