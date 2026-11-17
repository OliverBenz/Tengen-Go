#include "boardComparison.hpp"

#include <optional>

namespace tengen::app {

BoardComparison compareBoards(const Board& game, const Board& physical, const Player toMove) {
	using enum BoardComparison::Result;
	if (game.size() != physical.size()) {
		return {.result = Differs, .placed = {}};
	}

	std::optional<Coord> placed;
	for (unsigned x = 0; x < game.size(); ++x) {
		for (unsigned y = 0; y < game.size(); ++y) {
			const Coord point{x, y};
			if (physical.get(point) == game.get(point)) {
				continue;
			}

			const bool isPlacement = game.isEmpty(point) && physical.get(point) == toStone(toMove);
			if (!isPlacement || placed) {
				return {.result = Differs, .placed = {}};
			}
			placed = point;
		}
	}
	return placed ? BoardComparison{.result = StonePlaced, .placed = *placed} : BoardComparison{.result = Same, .placed = {}};
}

} // namespace tengen::app
