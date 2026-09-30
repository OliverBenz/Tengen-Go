#pragma once

#include <QColor>
#include <QImage>
#include <QStringList>

namespace tengen::gui {

inline constexpr QColor PLAIN_BOARD_COLOUR{220, 179, 92}; //!< Board colour when no texture is used.

//! Paths of all board background textures shipped in the board_textures resource folder, sorted by name.
QStringList boardTexturePaths();

//! Centre square of a board texture, as the board shows it. Null if unreadable.
QImage loadBoardTexture(const QString& path);

} // namespace tengen::gui
