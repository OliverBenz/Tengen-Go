#pragma once

#include <QPolygonF>
#include <QTransform>

class QPainter;

namespace tengen::labeler {

//! Maps grid coordinates (col, row), from 0 to size - 1, onto the photo whose 4 grid corners are given, in any order.
QTransform gridToPhoto(QPolygonF corners, unsigned size);

//! The size x size grid lines; `gridToTarget` maps grid coordinates (col, row) into the painter's coordinates.
//! Drawn over the photo, they show a wrong board size or a misplaced corner as lines beside the board's own.
void drawGrid(QPainter& painter, const QTransform& gridToTarget, unsigned size);

} // namespace tengen::labeler
