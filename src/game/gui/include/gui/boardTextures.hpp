#pragma once

#include <QColor>
#include <QStringList>

namespace tengen::gui {

inline constexpr QColor PLAIN_BOARD_COLOUR{220, 179, 92}; //!< Board colour when no texture is used.

//! Paths of all board background textures shipped in the board_textures resource folder, sorted by name.
QStringList boardTexturePaths();

} // namespace tengen::gui
