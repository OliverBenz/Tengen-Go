#pragma once

#include "model/gameRules.hpp"
#include "model/player.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <unordered_set>

namespace tengen {

//! Remembers the positions of a game and applies the ko rule to positions a move would create.
//! Positions are identified by the hash of the board and the player to move.
class PositionHistory {
public:
	explicit PositionHistory(Ko rule);

	bool allows(uint64_t boardHash, Player toMove) const; //!< False if reaching this position breaks the ko rule.
	void record(uint64_t boardHash, Player toMove);       //!< Called for the start position and after every move and pass.

private:
	Ko m_rule;

	// We use optional because 0 is a real hash.
	std::optional<uint64_t> m_previousBoard{};                  //!< Board hash before the last move. Simple ko bans returning to it.
	std::optional<uint64_t> m_currentBoard{};                   //!< Board hash at the current position.
	std::array<std::unordered_set<uint64_t>, 2> m_seenBoards{}; //!< Every board so far, one set per player to move.
};

} // namespace tengen
