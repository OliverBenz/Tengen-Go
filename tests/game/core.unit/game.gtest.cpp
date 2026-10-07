#include "core/game.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace tengen::gtest {
namespace {

//! Collects what the Game signalled, and in which order.
class StateRecorder : public IGameStateListener {
public:
	enum class Call { Start,
	                  Delta,
	                  End };

	void onGameStart(const GameConfig& config) override {
		calls.push_back(Call::Start);
		configs.push_back(config);
	}
	void onGameDelta(const GameDelta& delta) override {
		calls.push_back(Call::Delta);
		deltas.push_back(delta);
	}
	void onGameEnd(const GameResult& result) override {
		calls.push_back(Call::End);
		results.push_back(result);
	}

	std::vector<Call> calls;
	std::vector<GameConfig> configs;
	std::vector<GameDelta> deltas;
	std::vector<GameResult> results;
};
using Call = StateRecorder::Call;

//! A 9x9 game under Japanese rules.
GameConfig config() {
	return GameConfig{.boardSize = 9u, .rules = fromRuleSet(RuleSet::Japanese)};
}

//! Runs the game on its own thread, handles every event pushed so far, then stops it.
void runToEnd(Game& game) {
	std::thread gameThread([&] { game.run(); });
	game.pushEvent(ShutdownEvent{});
	gameThread.join();
}

} // namespace

//! A ko fought through the Game: the deltas carry every capture, and the ko rule holds back the immediate retake.
TEST(Game, BoardUpdate) {
	StateRecorder recorder;
	Game game(config());
	game.subscribeState(&recorder);
	game.pushEvent(StartEvent{});

	// Setup: black (1,2) is left with its last liberty at (1,1).
	game.pushEvent(PutStoneEvent{Player::Black, {0u, 1u}});
	game.pushEvent(PutStoneEvent{Player::White, {0u, 2u}});
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 0u}});
	game.pushEvent(PutStoneEvent{Player::White, {1u, 3u}});
	game.pushEvent(PutStoneEvent{Player::Black, {2u, 1u}});
	game.pushEvent(PutStoneEvent{Player::White, {2u, 2u}});
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 2u}});

	game.pushEvent(PutStoneEvent{Player::White, {1u, 1u}}); // White takes
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 2u}}); // Black cannot take (repeating board state)
	game.pushEvent(PutStoneEvent{Player::Black, {5u, 5u}}); // Black plays somewhere else
	game.pushEvent(PutStoneEvent{Player::White, {5u, 6u}}); // White plays somewhere else
	game.pushEvent(PutStoneEvent{Player::Black, {1u, 2u}}); // Black takes back

	runToEnd(game);
	game.unsubscribeState(&recorder);

	// Every move but the refused retake got through.
	ASSERT_EQ(recorder.deltas.size(), 11u);

	// The setup captures nothing.
	for (std::size_t i = 0u; i < 7u; ++i) {
		EXPECT_TRUE(recorder.deltas[i].captures.empty());
	}

	// White takes the ko.
	const auto& take = recorder.deltas[7];
	EXPECT_EQ(take.player, Player::White);
	ASSERT_EQ(take.captures.size(), 1u);
	EXPECT_EQ(take.captures[0].x, 1u);
	EXPECT_EQ(take.captures[0].y, 2u);

	// The refused retake left black to move, so black's move elsewhere is the next delta.
	const auto& elsewhere = recorder.deltas[8];
	EXPECT_EQ(elsewhere.moveId, 9u);
	EXPECT_EQ(elsewhere.player, Player::Black);
	ASSERT_TRUE(elsewhere.coord.has_value());
	EXPECT_EQ(elsewhere.coord->x, 5u);
	EXPECT_EQ(elsewhere.coord->y, 5u);

	// After the exchange black may take back.
	const auto& retake = recorder.deltas[10];
	EXPECT_EQ(retake.player, Player::Black);
	ASSERT_TRUE(retake.coord.has_value());
	EXPECT_EQ(retake.coord->x, 1u);
	EXPECT_EQ(retake.coord->y, 2u);
	ASSERT_EQ(retake.captures.size(), 1u);
	EXPECT_EQ(retake.captures[0].x, 1u);
	EXPECT_EQ(retake.captures[0].y, 1u);
}

//! The start carries the config, so listeners learn how the game is played. A second start changes nothing.
TEST(Game, SignalsStartWithConfig) {
	GameRules rules = fromRuleSet(RuleSet::Chinese);
	rules.komi      = 0.5f;

	StateRecorder recorder;
	Game game(GameConfig{.boardSize = 13u, .rules = rules});
	game.subscribeState(&recorder);

	game.pushEvent(StartEvent{});
	game.pushEvent(StartEvent{});
	runToEnd(game);
	game.unsubscribeState(&recorder);

	ASSERT_EQ(recorder.calls, std::vector<Call>{Call::Start});
	ASSERT_EQ(recorder.configs.size(), 1u);

	EXPECT_EQ(recorder.configs[0].boardSize, 13u);
	EXPECT_EQ(recorder.configs[0].rules.scoringMethod, Scoring::Area);
	EXPECT_EQ(recorder.configs[0].rules.komi, 0.5f);
}

//! The game can be set up early: nothing gets through until it is started.
TEST(Game, RejectsEventsBeforeStart) {
	StateRecorder recorder;
	Game game(config());
	game.subscribeState(&recorder);

	game.pushEvent(PutStoneEvent{Player::Black, {3u, 3u}});
	game.pushEvent(PassEvent{Player::Black});
	game.pushEvent(ResignEvent{Player::Black});
	game.pushEvent(StartEvent{});
	game.pushEvent(PutStoneEvent{Player::Black, {4u, 4u}});
	runToEnd(game);
	game.unsubscribeState(&recorder);

	ASSERT_EQ(recorder.calls, (std::vector<Call>{Call::Start, Call::Delta}));
	ASSERT_EQ(recorder.deltas.size(), 1u);

	EXPECT_EQ(recorder.deltas[0].moveId, 1u);
	ASSERT_TRUE(recorder.deltas[0].coord.has_value());
	EXPECT_EQ(recorder.deltas[0].coord->x, 4u);
	EXPECT_EQ(recorder.deltas[0].coord->y, 4u);
}

TEST(Game, RejectsEventsOutOfTurn) {
	StateRecorder recorder;
	Game game(config());
	game.subscribeState(&recorder);

	game.pushEvent(StartEvent{});
	game.pushEvent(PutStoneEvent{Player::Black, {3u, 3u}});
	game.pushEvent(PutStoneEvent{Player::Black, {4u, 4u}}); // Second click: black is not to move anymore.
	game.pushEvent(PassEvent{Player::Black});               // Neither is passing out of turn.
	game.pushEvent(PutStoneEvent{Player::White, {4u, 4u}});
	runToEnd(game);
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

TEST(Game, SignalsEndAfterTwoPasses) {
	StateRecorder recorder;
	Game game(config());
	game.subscribeState(&recorder);

	game.pushEvent(StartEvent{});
	game.pushEvent(PassEvent{Player::Black});
	game.pushEvent(PassEvent{Player::White});
	game.pushEvent(PutStoneEvent{Player::Black, {3u, 3u}});
	game.pushEvent(ResignEvent{Player::Black});
	runToEnd(game);
	game.unsubscribeState(&recorder);

	ASSERT_EQ(recorder.calls, (std::vector<Call>{Call::Start, Call::Delta, Call::Delta, Call::End}));
	ASSERT_EQ(recorder.deltas.size(), 2u);
	ASSERT_EQ(recorder.results.size(), 1u);

	EXPECT_EQ(recorder.deltas[1].action, GameAction::Pass);
	EXPECT_FALSE(recorder.results[0].winner.has_value());
	EXPECT_EQ(recorder.results[0].reason, EndReason::Counting);
}

TEST(Game, SignalsEndAfterResignOutOfTurn) {
	StateRecorder recorder;
	Game game(config());
	game.subscribeState(&recorder);

	game.pushEvent(StartEvent{});
	game.pushEvent(PutStoneEvent{Player::Black, {3u, 3u}});
	game.pushEvent(ResignEvent{Player::Black}); // White is to move.
	runToEnd(game);
	game.unsubscribeState(&recorder);

	ASSERT_EQ(recorder.calls, (std::vector<Call>{Call::Start, Call::Delta, Call::Delta, Call::End}));
	ASSERT_EQ(recorder.deltas.size(), 2u);
	ASSERT_EQ(recorder.results.size(), 1u);

	EXPECT_EQ(recorder.deltas[1].action, GameAction::Resign);
	EXPECT_EQ(recorder.deltas[1].player, Player::Black);
	EXPECT_EQ(recorder.deltas[1].nextPlayer, Player::White);
	EXPECT_EQ(recorder.results[0].winner, Player::White);
	EXPECT_EQ(recorder.results[0].reason, EndReason::Resignation);
}

} // namespace tengen::gtest
