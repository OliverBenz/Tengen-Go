#include "grid.hpp"

#include <QColor>
#include <QPainter>
#include <QPen>
#include <QPointF>

#include <algorithm>
#include <cmath>
#include <numeric>

namespace tengen::labeler {

static constexpr QColor LINE_COLOUR{255, 0, 255, 160}; //!< Unlike the board's lines, the wood and the corner markers.

QTransform gridToPhoto(QPolygonF corners, const unsigned size) {
	// Clockwise on screen, like rectify.order_clockwise() in tools/stoneClassifier, so both see the board the same way.
	const QPointF centre = std::accumulate(corners.begin(), corners.end(), QPointF()) / 4.0;
	std::ranges::sort(corners, {}, [centre](const QPointF corner) { return std::atan2(corner.y() - centre.y(), corner.x() - centre.x()); });

	const double last = size - 1.0;
	const QPolygonF grid{QPointF(0.0, 0.0), QPointF(last, 0.0), QPointF(last, last), QPointF(0.0, last)};
	QTransform transform;
	QTransform::quadToQuad(grid, corners, transform);
	return transform;
}

void drawGrid(QPainter& painter, const QTransform& gridToTarget, const unsigned size) {
	painter.setPen(QPen(LINE_COLOUR, 1.0));
	const double last = size - 1.0;
	for (unsigned line = 0; line < size; ++line) {
		const double at = line;
		painter.drawLine(gridToTarget.map(QPointF(0.0, at)), gridToTarget.map(QPointF(last, at)));
		painter.drawLine(gridToTarget.map(QPointF(at, 0.0)), gridToTarget.map(QPointF(at, last)));
	}
}

} // namespace tengen::labeler
