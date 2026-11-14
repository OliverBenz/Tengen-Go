#pragma once

#include "geometryFile.hpp"

#include <QImage>

namespace tengen::labeler {

//! The photo seen from straight above: a square grid with `spacing` px between its lines and one spacing of margin,
//! so intersection (row, col) lands at ((col + 1) * spacing, (row + 1) * spacing).
QImage rectify(const QImage& photo, const Geometry& geometry, int spacing);

} // namespace tengen::labeler
