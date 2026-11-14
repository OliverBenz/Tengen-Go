#include "boardFile.hpp"

#include <cassert>
#include <fstream>

namespace tengen::labeler {

std::filesystem::path boardFile(const std::filesystem::path& folder) {
	return folder / "board.txt";
}

bool writeBoard(const std::filesystem::path& file, const unsigned boardSize, const std::vector<Stone>& stones) {
	assert(stones.size() == std::size_t{boardSize} * boardSize);

	std::ofstream out(file);
	for (std::size_t i = 0; i < stones.size(); ++i) {
		out << static_cast<char>(stones[i]);
		if ((i + 1) % boardSize == 0) {
			out << '\n';
		}
	}
	out.close();
	return !out.fail();
}

} // namespace tengen::labeler
