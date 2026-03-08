#pragma once

#include "model/board.hpp"

#include <filesystem>

namespace tengen {
//! dotBW .. Text using '.' for empty 'B' for black and 'W' for white. Each board row in new line. Whitespace is ignored.
//! e.g 3x3 board  W . B
//                 W B .
//                 B . W
enum class SerializeFormat { dotBW };


/*! Read a board state from a single file.
 *  \param [in]  filePath The file containing the board information in the specified format.
 *  \param [out] board    Board information as specified in the file.
 *  \param [in]  format   The formt in which the board state is serialized in the given file.
 *  \returns     True on success.
 */
bool readBoard(std::filesystem::path testFile, Board& outBoard, SerializeFormat format = SerializeFormat::dotBW);

} // namespace tengen
