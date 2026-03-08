#include "core/serializer.hpp"

#include <fstream>
#include <algorithm>
#include <iostream>
#include <string>
#include <cassert>

namespace tengen {

static bool readDotBWBoard(std::filesystem::path testFile, Board& outBoard) {
	assert(testFile.extension().string() == ".txt"); // Our TXT testfiles must be stored as such.

	std::ifstream file(testFile);
	if (!file.is_open()) {
		std::cerr << "[Serializer] Could not open file: " << testFile << '\n';
		return false;
	}

	// Get first line, check board size.
	std::string line;
	std::getline(file, line);
	line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

	if (!(line.size() == 9 || line.size() == 13 || line.size() == 19)) {
		std::cerr << "[Serializer] Invalid board size: " << line.size() << '\n';
		return false;
	}

	Board board(line.size());
	unsigned y = 0u; //!< Board y coordinate.
	do {
		line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

		assert(y < board.size());
		if (line.size() != board.size()) {
			std::cerr << "[Serializer] Invalid test file format.\n";
			return false; // Invalid test file format.
		}

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
				std::cerr << "[Serializer] Invalid test file format.\n";
				return false; // Invalid test file format.
			}
			++x;
		}

		++y;
		assert(x == board.size());
	} while(std::getline(file, line));
	
	outBoard = std::move(board);
	return true;

}

bool readBoard(std::filesystem::path testFile, Board& outBoard, SerializeFormat format) {
	switch(format) {
	case SerializeFormat::dotBW:
		return readDotBWBoard(testFile, outBoard);
	}
	return false;
}

}
