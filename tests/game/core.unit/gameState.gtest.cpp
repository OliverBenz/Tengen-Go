#include "core/gameState.hpp"

#include <gtest/gtest.h>

namespace tengen::gtest {

TEST(GameState, TwoConsecutivePassesEndGame) {
	GameState state(9u, fromRuleSet(RuleSet::Japanese));

	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.isActive());

	EXPECT_TRUE(state.pass(Player::White));
	EXPECT_FALSE(state.isActive());
}

TEST(GameState, StoneResetsConsecutivePasses) {
	GameState state(9u, fromRuleSet(RuleSet::Japanese));

	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.place(Player::White, {3u, 3u}).has_value());
	EXPECT_TRUE(state.pass(Player::Black));
	EXPECT_TRUE(state.isActive());

	EXPECT_TRUE(state.pass(Player::White));
	EXPECT_FALSE(state.isActive());
}

TEST(GameState, ResignEndsGame) {
	GameState state(9u, fromRuleSet(RuleSet::Japanese));

	EXPECT_TRUE(state.resign());
	EXPECT_FALSE(state.isActive());
	EXPECT_FALSE(state.resign());
}

TEST(GameState, RejectsMovesAfterGameEnded) {
	GameState state(9u, fromRuleSet(RuleSet::Japanese));
	ASSERT_TRUE(state.pass(Player::Black));
	ASSERT_TRUE(state.pass(Player::White));

	const auto moveId = state.position().moveId;
	EXPECT_FALSE(state.place(Player::Black, {3u, 3u}).has_value());
	EXPECT_FALSE(state.pass(Player::Black));
	EXPECT_FALSE(state.resign());
	EXPECT_EQ(state.position().moveId, moveId);
}

TEST(GameState, RejectsMovesOutOfTurn) {
	GameState state(9u, fromRuleSet(RuleSet::Japanese));

	EXPECT_FALSE(state.place(Player::White, {3u, 3u}).has_value());
	EXPECT_FALSE(state.pass(Player::White));
	EXPECT_EQ(state.position().currentPlayer, Player::Black);
	EXPECT_EQ(state.position().moveId, 0u);
	EXPECT_TRUE(state.isActive());
}

} // namespace tengen::gtest
