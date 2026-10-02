#pragma once

namespace tengen {

//! The supported standard rulesets for the game.
enum class RuleSet {
	Japanese,
	Chinese,
	Korean
};

//! How we count the scoring at the end of the game.
enum class Scoring {
	Territory, //!< Surrounded empty points - captured prisoners.
	Area       //!< Surrounded empty points + living stones on board.
};

//! The Ko rule we use.
enum class Ko {
	Simple,      //!< Bans only immediately recapturing the stone just captured.
	Situational, //!< Bans recreating a (board position, player-to-move) pair seen earlier in this game.
	Positional   //!< Bans recreating a board position earlier seen in this game.
};

struct GameRules {
	Scoring scoringMethod;
	Ko koRule;
	float komi;
	bool suicideLegal;
};
GameRules fromRuleSet(RuleSet ruleSet); //!< Get the game rules given some standard ruleset.

} // namespace tengen
