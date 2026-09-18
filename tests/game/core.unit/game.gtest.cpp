#include "core/game.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace tengen::gtest {
namespace {

//! Collects what the Game actually accepted.
class DeltaRecorder : public IGameStateListener {
public:
	void onGameDelta(const GameDelta& delta) override {
		deltas.push_back(delta);
	}

	std::vector<GameDelta> deltas;
};

} // namespace

// TODO: Verify board state after every place
TEST(Game, BoardUpdate) {
	Game game(9u);
	std::thread gameThread([&] { game.run(); });

	// Setup ?koseki?
	game.pushEvent(PutStoneEvent{Player::Black, {0u, 1u}});
	game.pushEvent(PutStoneEvent{Player::White, {0u, 2u}});
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 0u}});
	game.pushEvent(PutStoneEvent{Player::White, {1u, 3u}});
	game.pushEvent(PutStoneEvent{Player::Black, {2u, 1u}});
	game.pushEvent(PutStoneEvent{Player::White, {2u, 2u}});
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 2u}});

	// White takes
	game.pushEvent(PutStoneEvent{Player::White, {1u, 1u}});

	// Black cannot take (repeating board state)
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 2u}});

	// Black plays somewhere else
	game.pushEvent(PutStoneEvent{Player::White, {5u, 5u}});

	// White plays somewhere else
	game.pushEvent(PutStoneEvent{Player::Black, {5u, 6u}});

	// Black takes back
	game.pushEvent(PutStoneEvent{Player::White, {1u, 2u}});


	game.pushEvent(ShutdownEvent{});
	gameThread.join();
}

//! A session against a bot leans on this: however fast the user clicks, only the player to move gets
//! his event through, so he can never place two stones in a row.
TEST(Game, RejectsEventsOutOfTurn) {
	DeltaRecorder recorder;
	Game game(9u);
	game.subscribeState(&recorder);
	std::thread gameThread([&] { game.run(); });

	game.pushEvent(PutStoneEvent{Player::Black, {3u, 3u}});
	game.pushEvent(PutStoneEvent{Player::Black, {4u, 4u}}); // Second click: black is not to move anymore.
	game.pushEvent(PassEvent{Player::Black});               // Neither is passing out of turn.
	game.pushEvent(PutStoneEvent{Player::White, {4u, 4u}});

	game.pushEvent(ShutdownEvent{});
	gameThread.join();
	game.unsubscribeState(&recorder);

	ASSERT_EQ(recorder.deltas.size(), 2u);
	EXPECT_EQ(recorder.deltas[0].player, Player::Black);
	EXPECT_EQ(recorder.deltas[0].nextPlayer, Player::White);
	ASSERT_TRUE(recorder.deltas[0].coord.has_value());
	EXPECT_EQ(recorder.deltas[0].coord->x, 3u);
	EXPECT_EQ(recorder.deltas[0].coord->y, 3u);

	EXPECT_EQ(recorder.deltas[1].player, Player::White);
	EXPECT_EQ(recorder.deltas[1].nextPlayer, Player::Black);
	ASSERT_TRUE(recorder.deltas[1].coord.has_value());
	EXPECT_EQ(recorder.deltas[1].coord->x, 4u);
	EXPECT_EQ(recorder.deltas[1].coord->y, 4u);
}

} // namespace tengen::gtest
