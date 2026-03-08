#include "testFileReader.hpp"

#include <fstream>
#include <string>
#include <cassert>

namespace tengen::vision {
namespace gtest {

bool readTestBoard(std::filesystem::path testFile, Board& outBoard) {
	assert(testFile.extension().string() == ".txt"); // Our TXT testfiles must be stored as such.

	std::ifstream file(testFile);
	if (!file.is_open()) {
		return false;
	}

	// Get first line, check board size.
	std::string line;
	std::getline(file, line);
	if (!(line.size() == 9 || line.size() == 13 || line.size() == 19)) {
		return false;
	}

	Board board(line.size());
	auto y = static_cast<unsigned>(board.size()); //!< Board y coordinate. Top to bottom parse.
	do {
		assert(y >= 1u);
		if (line.size() != board.size()) {
			return false; // Invalid test file format.
		}

		--y;
		unsigned x = 0u;
		for (const auto token : line) {
			switch(token) {
			case '.':
				break;
			case 'B':
				board.place({x, y}, Board::Stone::Black);
				break;
			case 'W':
				board.place({x, y}, Board::Stone::White);
				break;
			default:
				return false; // Invalid test file format.
			}
			++x;
		}

		assert(x == board.size());
	} while(std::getline(file, line));
	
	outBoard = std::move(board);
	return true;
}

}
}
