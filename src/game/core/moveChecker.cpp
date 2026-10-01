#include "core/moveChecker.hpp"

#include <array>
#include <concepts>
#include <utility>

namespace tengen {

namespace {

//! A connected group of stones of one colour.
struct Group {
	std::vector<Coord> stones;
	std::size_t liberties{0};
};

bool inBounds(const Board& board, const Coord c) {
	return c.x < board.size() && c.y < board.size();
}

template <std::invocable<Coord> Visitor>
void forEachNeighbour(const Board& board, const Coord c, Visitor&& visit) {
	static constexpr std::array<std::pair<int, int>, 4> offsets{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

	const auto size = static_cast<int>(board.size());
	for (const auto [dx, dy]: offsets) {
		const int x = static_cast<int>(c.x) + dx;
		const int y = static_cast<int>(c.y) + dy;
		if (x >= 0 && y >= 0 && x < size && y < size) {
			visit(Coord{static_cast<unsigned>(x), static_cast<unsigned>(y)});
		}
	}
}

//! The group containing the stone at start, which must not be empty.
Group findGroup(const Board& board, const Coord start) {
	const auto colour = board.get(start);
	const auto size   = board.size();
	const auto index  = [size](const Coord c) { return c.y * size + c.x; };

	std::vector<bool> inGroup(size * size, false);
	std::vector<bool> isLiberty(size * size, false);

	Group group;
	std::vector<Coord> pending{start};
	inGroup[index(start)] = true;

	while (!pending.empty()) {
		const auto stone = pending.back();
		pending.pop_back();
		group.stones.push_back(stone);

		forEachNeighbour(board, stone, [&](const Coord neighbour) {
			const auto value = board.get(neighbour);
			if (value == colour && !inGroup[index(neighbour)]) {
				inGroup[index(neighbour)] = true;
				pending.push_back(neighbour);
			} else if (value == Board::Stone::Empty && !isLiberty[index(neighbour)]) {
				isLiberty[index(neighbour)] = true;
				++group.liberties;
			}
		});
	}
	return group;
}

void removeStones(Board& board, const std::vector<Coord>& stones, std::vector<Coord>& removed) {
	removed.reserve(stones.size());

	for (const auto stone: stones) {
		board.remove(stone);
		removed.push_back(stone);
	}
}

} // namespace

std::optional<Placement> playStone(const Board& board, const Player player, const Coord c, const bool suicideLegal) {
	if (!inBounds(board, c) || !board.isEmpty(c)) {
		return std::nullopt;
	}

	Placement placement{.board = board, .captured = {}, .selfCaptured = {}};
	placement.board.place(c, toStone(player));

	// Check if we killed someone
	const auto enemyStone = toStone(opponent(player));
	forEachNeighbour(placement.board, c, [&](const Coord neighbour) {
		if (placement.board.get(neighbour) != enemyStone) {
			return;
		}

		// If we have an enemy neighbour, check if his group still has liberties.
		// TODO: We could just check hasLiberties(placement.board, neighbour). If enemy group has even 1, then not dead.
		const auto group = findGroup(placement.board, neighbour);
		if (group.liberties == 0) {
			removeStones(placement.board, group.stones, placement.captured);
		}
	});

	// Check suicide
	const auto own = findGroup(placement.board, c);
	if (own.liberties > 0) {
		return placement;
	}

	// A lone stone taking itself off leaves the board unchanged: a pass in disguise.
	if (!suicideLegal || own.stones.size() == 1) {
		return std::nullopt;
	}

	// We killed ourself :(
	removeStones(placement.board, own.stones, placement.selfCaptured);
	return placement;
}

std::size_t computeGroupLiberties(const Board& board, const Coord c, const Player player) {
	if (!inBounds(board, c)) {
		return 0;
	}

	Board withStone = board;
	withStone.place(c, toStone(player)); // Keeps an existing stone.
	if (withStone.get(c) != toStone(player)) {
		return 0;
	}
	return findGroup(withStone, c).liberties;
}

} // namespace tengen
