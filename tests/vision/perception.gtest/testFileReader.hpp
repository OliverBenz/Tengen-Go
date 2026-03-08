#pragma once

#include "model/board.hpp"

#include <filesystem>

namespace tengen::vision {
namespace gtest {

/*! Read a board state from a test file.
 *  \param [in]  testFile The TXT file containing the board information in .BW format.
 *  \param [out] board    Board information as specified in the file.
 *  \returns     True on success.
 */
bool readTestBoard(std::filesystem::path testFile, Board& outBoard);

}
}
