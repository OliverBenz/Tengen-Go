#include "core/moveChecker.hpp"
#include "model/board.hpp"

#include <gtest/gtest.h>

namespace tengen::gtest {

// Liberties of single stones at all board positions
TEST(MoveChecker, ComputeConnectedLiberties_Single) {
	Board board(9u);
	// In corner
	EXPECT_EQ(computeGroupLiberties(board, {0u, 0u}, Player::Black), 2u);
	EXPECT_EQ(computeGroupLiberties(board, {8u, 8u}, Player::Black), 2u);
	EXPECT_EQ(computeGroupLiberties(board, {0u, 8u}, Player::Black), 2u);
	EXPECT_EQ(computeGroupLiberties(board, {8u, 0u}, Player::Black), 2u);
	// At border
	for (unsigned i = 1; i != 8; ++i) {
		EXPECT_EQ(computeGroupLiberties(board, {i, 0u}, Player::Black), 3u);
		EXPECT_EQ(computeGroupLiberties(board, {i, 8u}, Player::Black), 3u);
	}
	for (unsigned j = 1; j != 8; ++j) {
		EXPECT_EQ(computeGroupLiberties(board, {0u, j}, Player::Black), 3u);
		EXPECT_EQ(computeGroupLiberties(board, {8u, j}, Player::Black), 3u);
	}
	// No borders
	for (unsigned i = 1; i != 8; ++i) {
		for (unsigned j = 1; j != 8; ++j) {
			EXPECT_EQ(computeGroupLiberties(board, {i, j}, Player::Black), 4u);
		}
	}
}

// Liberties of groups not touching borders
TEST(MoveChecker, ComputeConnectedLiberties_Center) {
	{
		Board board(9u);

		board.place({4u, 3u}, Board::Stone::Black);
		board.place({4u, 4u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {4u, 3u}, Player::Black), 6u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 4u}, Player::Black), 6u);
	}
	{
		Board board(9u);

		board.place({4u, 3u}, Board::Stone::Black);
		board.place({4u, 4u}, Board::Stone::Black);
		board.place({4u, 5u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {4u, 3u}, Player::Black), 8u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 4u}, Player::Black), 8u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 5u}, Player::Black), 8u);
	}
	{
		Board board(9u);

		board.place({4u, 3u}, Board::Stone::Black);
		board.place({4u, 4u}, Board::Stone::Black);
		board.place({5u, 4u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {4u, 3u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 4u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {5u, 4u}, Player::Black), 7u);
	}
	{
		Board board(9u);

		board.place({4u, 3u}, Board::Stone::Black);
		board.place({4u, 4u}, Board::Stone::Black);
		board.place({4u, 5u}, Board::Stone::Black);
		board.place({5u, 5u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {4u, 3u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 4u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 5u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {5u, 5u}, Player::Black), 9u);
	}
	{
		Board board(9u);

		board.place({4u, 3u}, Board::Stone::Black);
		board.place({4u, 4u}, Board::Stone::Black);
		board.place({4u, 5u}, Board::Stone::Black);

		board.place({5u, 3u}, Board::Stone::Black);
		board.place({5u, 5u}, Board::Stone::Black);

		board.place({6u, 4u}, Board::Stone::Black);
		board.place({6u, 5u}, Board::Stone::Black);

		board.place({7u, 4u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {4u, 3u}, Player::Black), 13u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 4u}, Player::Black), 13u);
		EXPECT_EQ(computeGroupLiberties(board, {4u, 5u}, Player::Black), 13u);

		EXPECT_EQ(computeGroupLiberties(board, {5u, 3u}, Player::Black), 13u);
		EXPECT_EQ(computeGroupLiberties(board, {5u, 5u}, Player::Black), 13u);

		EXPECT_EQ(computeGroupLiberties(board, {6u, 4u}, Player::Black), 13u);
		EXPECT_EQ(computeGroupLiberties(board, {6u, 5u}, Player::Black), 13u);

		EXPECT_EQ(computeGroupLiberties(board, {7u, 4u}, Player::Black), 13u);
	}
}

// Liberties of groups touching borders and corners
TEST(MoveChecker, ComputeConnectedLiberties_Borders) {
	{
		Board board(9u);

		board.place({0u, 0u}, Board::Stone::Black);
		board.place({0u, 1u}, Board::Stone::Black);
		board.place({0u, 2u}, Board::Stone::Black);

		board.place({1u, 1u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {0u, 0u}, Player::Black), 4u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 1u}, Player::Black), 4u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 2u}, Player::Black), 4u);

		EXPECT_EQ(computeGroupLiberties(board, {1u, 1u}, Player::Black), 4u);
	}
	{
		Board board(9u);

		board.place({0u, 0u}, Board::Stone::Black);
		board.place({0u, 1u}, Board::Stone::Black);
		board.place({0u, 2u}, Board::Stone::Black);

		board.place({1u, 1u}, Board::Stone::Black);

		board.place({2u, 0u}, Board::Stone::Black);
		board.place({2u, 1u}, Board::Stone::Black);
		board.place({2u, 2u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {0u, 0u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 1u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 2u}, Player::Black), 7u);

		EXPECT_EQ(computeGroupLiberties(board, {1u, 1u}, Player::Black), 7u);

		EXPECT_EQ(computeGroupLiberties(board, {2u, 0u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 1u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 2u}, Player::Black), 7u);
	}
	{
		Board board(9u);

		board.place({0u, 0u}, Board::Stone::Black);
		board.place({0u, 1u}, Board::Stone::Black);
		board.place({0u, 2u}, Board::Stone::Black);

		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);

		board.place({2u, 0u}, Board::Stone::Black);
		board.place({2u, 1u}, Board::Stone::Black);
		board.place({2u, 2u}, Board::Stone::Black);

		// Check liberties for each stone to check that full chain is found.
		EXPECT_EQ(computeGroupLiberties(board, {0u, 0u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 1u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {0u, 2u}, Player::Black), 7u);

		EXPECT_EQ(computeGroupLiberties(board, {1u, 0u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {1u, 2u}, Player::Black), 7u);

		EXPECT_EQ(computeGroupLiberties(board, {2u, 0u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 1u}, Player::Black), 7u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 2u}, Player::Black), 7u);
	}
	{
		Board board(9u);

		board.place({0u, 0u}, Board::Stone::Black);
		board.place({1u, 0u}, Board::Stone::Black);
		board.place({2u, 0u}, Board::Stone::Black);
		board.place({3u, 0u}, Board::Stone::Black);

		board.place({0u, 1u}, Board::Stone::Black);
		board.place({3u, 1u}, Board::Stone::Black);

		board.place({0u, 2u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);
		board.place({2u, 2u}, Board::Stone::Black);
		board.place({3u, 2u}, Board::Stone::Black);

		EXPECT_EQ(computeGroupLiberties(board, {0u, 0u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {1u, 0u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 0u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {3u, 0u}, Player::Black), 9u);

		EXPECT_EQ(computeGroupLiberties(board, {0u, 1u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {3u, 1u}, Player::Black), 9u);

		EXPECT_EQ(computeGroupLiberties(board, {0u, 2u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {1u, 2u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {2u, 2u}, Player::Black), 9u);
		EXPECT_EQ(computeGroupLiberties(board, {3u, 2u}, Player::Black), 9u);
	}
}

// TODO: Write tests to check we find the whole group and liberty count
TEST(MoveChecker, FindGroup) {
}

TEST(MoveChecker, Suicide) {
	{
		Board board(9u);

		board.place({0u, 1u}, Board::Stone::Black);
		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);

		// Legal move
		EXPECT_TRUE(playStone(board, Player::Black, {1u, 1u}, false).has_value());
		EXPECT_EQ(computeGroupLiberties(board, {1u, 1u}, Player::White), 1u);
	}

	{
		Board board(9u);

		board.place({0u, 1u}, Board::Stone::Black);
		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);
		board.place({2u, 1u}, Board::Stone::Black);

		// Suicide -> invalid move
		EXPECT_FALSE(playStone(board, Player::White, {1u, 1u}, false).has_value());
		EXPECT_EQ(computeGroupLiberties(board, {1u, 1u}, Player::White), 0u);
	}
	{
		Board board(9u);

		board.place({0u, 1u}, Board::Stone::Black);
		board.place({0u, 2u}, Board::Stone::Black);
		board.place({0u, 3u}, Board::Stone::Black);

		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 1u}, Board::Stone::White);
		board.place({1u, 2u}, Board::Stone::White);
		board.place({1u, 3u}, Board::Stone::Black);

		board.place({2u, 0u}, Board::Stone::Black);
		board.place({2u, 1u}, Board::Stone::White);
		board.place({2u, 2u}, Board::Stone::Black);
		board.place({2u, 3u}, Board::Stone::Black);

		board.place({3u, 0u}, Board::Stone::Black);
		board.place({3u, 2u}, Board::Stone::Black);

		board.place({4u, 1u}, Board::Stone::Black);

		// Suicide -> invalid move
		EXPECT_FALSE(playStone(board, Player::White, {3u, 1u}, false).has_value());
		EXPECT_EQ(computeGroupLiberties(board, {3u, 1u}, Player::White), 0u);

		// Now add white stones which would allow the same move to be a capture
		// Surround the rightmost black stone.
		board.place({4u, 0u}, Board::Stone::White);
		board.place({4u, 2u}, Board::Stone::White);
		board.place({5u, 1u}, Board::Stone::White);

		// Now we capture -> Move valid
		EXPECT_TRUE(playStone(board, Player::White, {3u, 1u}, false).has_value());
		EXPECT_EQ(computeGroupLiberties(board, {3u, 1u}, Player::White), 0u);
	}

	{
		Board board(9u);

		board.place({0u, 1u}, Board::Stone::Black);

		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);

		board.place({2u, 0u}, Board::Stone::White);
		board.place({2u, 1u}, Board::Stone::Black);
		board.place({2u, 2u}, Board::Stone::White);

		board.place({3u, 1u}, Board::Stone::White);

		// Captures -> valid move
		EXPECT_TRUE(playStone(board, Player::White, {1u, 1u}, false).has_value());
		EXPECT_EQ(computeGroupLiberties(board, {1u, 1u}, Player::White), 0u);
	}
}

TEST(MoveChecker, Kill) {
	{
		Board board(9u);

		board.place({0u, 0u}, Board::Stone::White);
		board.place({0u, 1u}, Board::Stone::Black);
		board.place({0u, 2u}, Board::Stone::White);

		board.place({1u, 0u}, Board::Stone::Black);
		board.place({1u, 2u}, Board::Stone::Black);
		board.place({1u, 3u}, Board::Stone::White);

		board.place({2u, 0u}, Board::Stone::White);
		board.place({2u, 1u}, Board::Stone::Black);
		board.place({2u, 2u}, Board::Stone::White);

		board.place({3u, 1u}, Board::Stone::White);

		const auto placement = playStone(board, Player::White, {1u, 1u}, false);
		ASSERT_TRUE(placement.has_value());
		EXPECT_EQ(placement->captured.size(), 4u);
		EXPECT_TRUE(placement->selfCaptured.empty());
		EXPECT_EQ(placement->board.get({1u, 1u}), Board::Stone::White);
		EXPECT_TRUE(placement->board.isEmpty({0u, 1u}));
		EXPECT_TRUE(placement->board.isEmpty({1u, 0u}));
		EXPECT_TRUE(placement->board.isEmpty({1u, 2u}));
		EXPECT_TRUE(placement->board.isEmpty({2u, 1u}));
	}
}

TEST(MoveChecker, RejectsOffBoardAndOccupied) {
	Board board(9u);
	board.place({4u, 4u}, Board::Stone::Black);

	EXPECT_FALSE(playStone(board, Player::White, {4u, 4u}, false).has_value());
	EXPECT_FALSE(playStone(board, Player::White, {9u, 0u}, false).has_value());
	EXPECT_FALSE(playStone(board, Player::White, {0u, 9u}, false).has_value());
}

TEST(MoveChecker, LegalSuicideRemovesOwnGroup) {
	Board board(9u);
	board.place({0u, 0u}, Board::Stone::White);
	board.place({1u, 0u}, Board::Stone::Black);
	board.place({1u, 1u}, Board::Stone::Black);
	board.place({0u, 2u}, Board::Stone::Black);

	EXPECT_FALSE(playStone(board, Player::White, {0u, 1u}, false).has_value());

	const auto placement = playStone(board, Player::White, {0u, 1u}, true);
	ASSERT_TRUE(placement.has_value());
	EXPECT_TRUE(placement->captured.empty());
	EXPECT_EQ(placement->selfCaptured.size(), 2u);
	EXPECT_TRUE(placement->board.isEmpty({0u, 0u}));
	EXPECT_TRUE(placement->board.isEmpty({0u, 1u}));
	EXPECT_EQ(placement->board.get({1u, 0u}), Board::Stone::Black);
}

TEST(MoveChecker, LoneStoneSuicideIsNeverLegal) {
	Board board(9u);
	board.place({0u, 1u}, Board::Stone::Black);
	board.place({1u, 0u}, Board::Stone::Black);
	board.place({1u, 2u}, Board::Stone::Black);
	board.place({2u, 1u}, Board::Stone::Black);

	EXPECT_FALSE(playStone(board, Player::White, {1u, 1u}, true).has_value());
}

} // namespace tengen::gtest
