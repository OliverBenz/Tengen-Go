#include "core/gameState.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

namespace tengen::gtest {

//! A 9x9 game under the given rules.
static GameConfig config(const GameRules& rules = fromRuleSet(RuleSet::Japanese)) {
	return GameConfig{.boardSize = 9u, .rules = rules};
}

TEST(GameState, RejectsUnsupportedBoardSize) {
	const GameConfig sevenBySeven{.boardSize = 7u, .rules = fromRuleSet(RuleSet::Japanese)};
	EXPECT_FALSE(isSupportedBoardSize(sevenBySeven.boardSize));
	EXPECT_THROW(GameState{sevenBySeven}, std::invalid_argument);
}

TEST(GameState, PlaysEverySupportedBoardSize) {
	for (const auto size: SUPPORTED_BOARD_SIZES) {
		EXPECT_TRUE(isSupportedBoardSize(size));
		EXPECT_NO_THROW(GameState(GameConfig{.boardSize = size, .rules = fromRuleSet(RuleSet::Japanese)})) << "Board size " << size;
	}
}

TEST(GameState, KeepsConfig) {
	GameRules rules = fromRuleSet(RuleSet::Chinese);
	rules.komi      = 0.5f;
	GameState state(GameConfig{.boardSize = 13u, .rules = rules});

	EXPECT_EQ(state.config().boardSize, 13u);
	EXPECT_EQ(state.config().rules.komi, 0.5f);
	EXPECT_EQ(state.position().board.size(), 13u);
}

TEST(GameState, RejectsMovesBeforeStart) {
	GameState state(config());
	EXPECT_FALSE(state.isActive());

	EXPECT_FALSE(state.place(Player::Black, {3u, 3u}).has_value());
	EXPECT_FALSE(state.pass(Player::Black));
	EXPECT_FALSE(state.resign(Player::Black));
	EXPECT_EQ(state.position().moveId, 0u);
	EXPECT_FALSE(state.result().has_value());

	ASSERT_TRUE(state.start());
	EXPECT_TRUE(state.isActive());
	EXPECT_TRUE(state.place(Player::Black, {3u, 3u}).has_value());
}

TEST(GameState, StartsOnlyOnce) {
	GameState state(config());

	EXPECT_TRUE(state.start());
	EXPECT_FALSE(state.start());
	EXPECT_TRUE(state.isActive());
}

TEST(GameState, TwoConsecutivePassesEndGame) {
	GameState state(config());
	ASSERT_TRUE(state.start());

	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.isActive());
	EXPECT_FALSE(state.result().has_value());

	EXPECT_TRUE(state.pass(Player::White));
	EXPECT_FALSE(state.isActive());

	// TODO: The board is not counted yet, so the result names no winner.
	ASSERT_TRUE(state.result().has_value());
	EXPECT_FALSE(state.result()->winner.has_value());
	EXPECT_EQ(state.result()->reason, EndReason::Counting);
}

TEST(GameState, StoneResetsConsecutivePasses) {
	GameState state(config());
	ASSERT_TRUE(state.start());

	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.place(Player::White, {3u, 3u}).has_value());
	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.isActive());

	EXPECT_TRUE(state.pass(Player::White));
	EXPECT_FALSE(state.isActive());
}

TEST(GameState, ResignEndsGame) {
	GameState state(config());
	ASSERT_TRUE(state.start());

	EXPECT_TRUE(state.resign(Player::Black));
	EXPECT_FALSE(state.isActive());
	EXPECT_FALSE(state.resign(Player::White));

	ASSERT_TRUE(state.result().has_value());
	EXPECT_EQ(state.result()->winner, Player::White);
	EXPECT_EQ(state.result()->reason, EndReason::Resignation);
}

TEST(GameState, ResignOutOfTurn) {
	GameState state(config());
	ASSERT_TRUE(state.start());
	ASSERT_TRUE(state.place(Player::Black, {3u, 3u}).has_value());

	EXPECT_TRUE(state.resign(Player::Black));

	ASSERT_TRUE(state.result().has_value());
	EXPECT_EQ(state.result()->winner, Player::White);
	EXPECT_EQ(state.result()->reason, EndReason::Resignation);
}

TEST(GameState, RejectsMovesAfterGameEnded) {
	GameState state(config());
	ASSERT_TRUE(state.start());
	ASSERT_TRUE(state.pass(Player::Black));
	ASSERT_TRUE(state.pass(Player::White));

	const auto moveId = state.position().moveId;
	EXPECT_FALSE(state.place(Player::Black, {3u, 3u}).has_value());
	EXPECT_FALSE(state.pass(Player::Black));
	EXPECT_FALSE(state.resign(Player::Black));
	EXPECT_FALSE(state.start());
	EXPECT_EQ(state.position().moveId, moveId);
	EXPECT_EQ(state.result()->reason, EndReason::Counting);
}

TEST(GameState, RejectsMovesOutOfTurn) {
	GameState state(config());
	ASSERT_TRUE(state.start());

	EXPECT_FALSE(state.place(Player::White, {3u, 3u}).has_value());
	EXPECT_FALSE(state.pass(Player::White));
	EXPECT_EQ(state.position().currentPlayer, Player::Black);
	EXPECT_EQ(state.position().moveId, 0u);
	EXPECT_TRUE(state.isActive());
}

//! Black surrounds a white stone at (1,1) and takes it with (2,1), which white could retake at once.
static std::optional<std::vector<Coord>> blackTakesKo(GameState& state) {
	EXPECT_TRUE(state.place(Player::Black, {1u, 0u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {2u, 0u}).has_value());
	EXPECT_TRUE(state.place(Player::Black, {0u, 1u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {3u, 1u}).has_value());
	EXPECT_TRUE(state.place(Player::Black, {1u, 2u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {2u, 2u}).has_value());
	EXPECT_TRUE(state.place(Player::Black, {8u, 8u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {1u, 1u}).has_value());
	return state.place(Player::Black, {2u, 1u});
}

TEST(GameState, KoRetakeAfterExchange) {
	for (const auto ko: {Ko::Simple, Ko::Situational, Ko::Positional}) {
		SCOPED_TRACE(static_cast<int>(ko));
		GameRules rules = fromRuleSet(RuleSet::Japanese);
		rules.koRule    = ko;
		GameState state(config(rules));
		ASSERT_TRUE(state.start());

		const auto captures = blackTakesKo(state);
		ASSERT_TRUE(captures.has_value());
		EXPECT_EQ(captures->size(), 1u);

		EXPECT_FALSE(state.place(Player::White, {1u, 1u}).has_value());
		EXPECT_EQ(state.position().currentPlayer, Player::White);

		ASSERT_TRUE(state.place(Player::White, {8u, 0u}).has_value());
		ASSERT_TRUE(state.place(Player::Black, {7u, 0u}).has_value());
		EXPECT_TRUE(state.place(Player::White, {1u, 1u}).has_value());
	}
}

//! Black closes in on the corner so that white at (0,1) leaves the white group at (0,0) without liberties.
static void prepareWhiteSuicide(GameState& state) {
	EXPECT_TRUE(state.place(Player::Black, {1u, 0u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {0u, 0u}).has_value());
	EXPECT_TRUE(state.place(Player::Black, {0u, 2u}).has_value());
	EXPECT_TRUE(state.place(Player::White, {8u, 8u}).has_value());
	EXPECT_TRUE(state.place(Player::Black, {1u, 1u}).has_value());
}

TEST(GameState, SuicideIllegal) {
	GameState state(config());
	ASSERT_TRUE(state.start());
	prepareWhiteSuicide(state);

	const auto moveId = state.position().moveId;
	EXPECT_FALSE(state.place(Player::White, {0u, 1u}).has_value());
	EXPECT_EQ(state.position().currentPlayer, Player::White);
	EXPECT_EQ(state.position().moveId, moveId);
}

TEST(GameState, SuicideLegal) {
	GameRules rules    = fromRuleSet(RuleSet::Japanese);
	rules.suicideLegal = true;
	GameState state(config(rules));
	ASSERT_TRUE(state.start());
	prepareWhiteSuicide(state);

	const auto removed = state.place(Player::White, {0u, 1u});
	ASSERT_TRUE(removed.has_value());
	EXPECT_EQ(removed->size(), 2u);
	EXPECT_TRUE(state.position().board.isEmpty({0u, 0u}));
	EXPECT_TRUE(state.position().board.isEmpty({0u, 1u}));
	EXPECT_EQ(state.position().currentPlayer, Player::Black);
}

} // namespace tengen::gtest
