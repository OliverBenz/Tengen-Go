#pragma once

#include <filesystem>
#include <vector>

namespace tengen::labeler {

//! Each value is its character in the dotBW format, see tests/vision/resources/README.md.
enum class Stone : char {
	empty = '.',
	black = 'B',
	white = 'W',
};

//! Where the stone layout shared by all photos of a folder goes: `<folder>/board.txt`.
std::filesystem::path boardFile(const std::filesystem::path& folder);

//! `stones` row by row, one line per row. False if the file cannot be written.
bool writeBoard(const std::filesystem::path& file, unsigned boardSize, const std::vector<Stone>& stones);

} // namespace tengen::labeler
