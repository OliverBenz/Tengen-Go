#include "network/nwEvents.hpp"

#include <gtest/gtest.h>

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace tengen::gtest {

TEST(GameNetMessages, ClientToMessage) {
	using nlohmann::json;

	EXPECT_EQ(json::parse(network::toMessage(network::ClientPutStone{.c = {1u, 2u}})), json({{"type", "put"}, {"x", 1u}, {"y", 2u}}));
	EXPECT_EQ(json::parse(network::toMessage(network::ClientPass{})), json({{"type", "pass"}}));
	EXPECT_EQ(json::parse(network::toMessage(network::ClientResign{})), json({{"type", "resign"}}));
	EXPECT_EQ(json::parse(network::toMessage(network::ClientChat{"hello"})), json({{"type", "chat"}, {"message", "hello"}}));
}

TEST(GameNetMessages, ClientFromMessageValid) {
	const auto put = network::fromClientMessage(R"({"type":"put","x":3,"y":4})");
	ASSERT_TRUE(put.has_value());
	ASSERT_TRUE(std::holds_alternative<network::ClientPutStone>(*put));
	const auto putEvent = std::get<network::ClientPutStone>(*put);
	EXPECT_EQ(putEvent.c.x, 3u);
	EXPECT_EQ(putEvent.c.y, 4u);

	const auto pass = network::fromClientMessage(R"({"type":"pass"})");
	ASSERT_TRUE(pass.has_value());
	EXPECT_TRUE(std::holds_alternative<network::ClientPass>(*pass));

	const auto resign = network::fromClientMessage(R"({"type":"resign"})");
	ASSERT_TRUE(resign.has_value());
	EXPECT_TRUE(std::holds_alternative<network::ClientResign>(*resign));

	const auto chat = network::fromClientMessage(R"({"type":"chat","message":"hello"})");
	ASSERT_TRUE(chat.has_value());
	ASSERT_TRUE(std::holds_alternative<network::ClientChat>(*chat));
	EXPECT_EQ(std::get<network::ClientChat>(*chat).message, "hello");
}

TEST(GameNetMessages, ClientFromMessageInvalid) {
	EXPECT_FALSE(network::fromClientMessage(R"({"type":"put","x":1})").has_value());
	EXPECT_FALSE(network::fromClientMessage(R"({"type":"put","x":"1","y":2})").has_value());
	EXPECT_FALSE(network::fromClientMessage(R"({"type":"chat"})").has_value());
	EXPECT_FALSE(network::fromClientMessage(R"({"type":"unknown"})").has_value());
	EXPECT_FALSE(network::fromClientMessage("not-json").has_value());
}

namespace {

//! A move by the player, with the opponent to move next.
GameDelta makeMove(const unsigned moveId, const GameAction action, const Player player, const std::optional<Coord> coord = std::nullopt,
                   std::vector<Coord> captures = {}) {
	return GameDelta{.moveId = moveId, .action = action, .player = player, .coord = coord, .captures = std::move(captures), .nextPlayer = opponent(player)};
}

//! Parses a server message as the given event. Nullopt if it is invalid or another event.
template <typename T>
std::optional<T> parseAs(const std::string& message) {
	const auto event = network::fromServerMessage(message);
	if (!event || !std::holds_alternative<T>(*event)) {
		return std::nullopt;
	}
	return std::get<T>(*event);
}

//! Serializes the event and parses it back.
template <typename T>
std::optional<T> roundTrip(const T& event) {
	return parseAs<T>(network::toMessage(event));
}

} // namespace

TEST(GameNetMessages, ServerToMessage) {
	using nlohmann::json;

	EXPECT_EQ(json::parse(network::toMessage(network::ServerSessionAssign{1u})), json({{"type", "session"}, {"sessionId", 1u}}));

	const GameRules rules{.scoringMethod = Scoring::Area, .koRule = Ko::Situational, .komi = 7.5f, .suicideLegal = true};
	EXPECT_EQ(json::parse(network::toMessage(network::ServerGameStart{GameConfig{.boardSize = 19u, .rules = rules}})),
	          json({{"type", "start"}, {"boardSize", 19u}, {"rules", {{"scoring", "area"}, {"ko", "situational"}, {"komi", 7.5}, {"suicide", true}}}}));

	EXPECT_EQ(json::parse(network::toMessage(
	                  network::ServerGameDelta{makeMove(42u, GameAction::Place, Player::Black, Coord{3u, 4u}, {Coord{1u, 2u}, Coord{5u, 6u}})})),
	          json({{"type", "delta"},
	                {"moveId", 42u},
	                {"action", "place"},
	                {"player", "black"},
	                {"next", "white"},
	                {"x", 3u},
	                {"y", 4u},
	                {"captures", json::array({json::array({1u, 2u}), json::array({5u, 6u})})}}));

	EXPECT_EQ(json::parse(network::toMessage(network::ServerGameDelta{makeMove(43u, GameAction::Pass, Player::White)})),
	          json({{"type", "delta"}, {"moveId", 43u}, {"action", "pass"}, {"player", "white"}, {"next", "black"}}));

	EXPECT_EQ(json::parse(network::toMessage(network::ServerGameDelta{makeMove(44u, GameAction::Resign, Player::Black)})),
	          json({{"type", "delta"}, {"moveId", 44u}, {"action", "resign"}, {"player", "black"}, {"next", "white"}}));

	EXPECT_EQ(json::parse(network::toMessage(network::ServerGameEnd{GameResult{.winner = Player::White, .reason = EndReason::Resignation}})),
	          json({{"type", "end"}, {"reason", "resignation"}, {"winner", "white"}}));

	EXPECT_EQ(json::parse(network::toMessage(network::ServerChat{Player::White, 0u, "hi"})),
	          json({{"type", "chat"}, {"player", "white"}, {"messageId", 0u}, {"message", "hi"}}));
}

TEST(GameNetMessages, ServerFromMessageValid) {
	const auto session = parseAs<network::ServerSessionAssign>(R"({"type":"session","sessionId":42})");
	ASSERT_TRUE(session.has_value());
	EXPECT_EQ(session->sessionId, 42u);

	const auto start =
	        parseAs<network::ServerGameStart>(R"({"type":"start","boardSize":13,"rules":{"scoring":"territory","ko":"simple","komi":6.5,"suicide":false}})");
	ASSERT_TRUE(start.has_value());
	EXPECT_EQ(start->config.boardSize, 13u);
	EXPECT_EQ(start->config.rules.scoringMethod, Scoring::Territory);
	EXPECT_EQ(start->config.rules.koRule, Ko::Simple);
	EXPECT_EQ(start->config.rules.komi, 6.5f);
	EXPECT_FALSE(start->config.rules.suicideLegal);

	const auto place = parseAs<network::ServerGameDelta>(
	        R"({"type":"delta","moveId":7,"action":"place","player":"black","next":"white","x":1,"y":2,"captures":[[3,4],[5,6]]})");
	ASSERT_TRUE(place.has_value());
	EXPECT_EQ(place->delta.moveId, 7u);
	EXPECT_EQ(place->delta.action, GameAction::Place);
	EXPECT_EQ(place->delta.player, Player::Black);
	EXPECT_EQ(place->delta.nextPlayer, Player::White);
	ASSERT_TRUE(place->delta.coord.has_value());
	EXPECT_EQ(place->delta.coord->x, 1u);
	EXPECT_EQ(place->delta.coord->y, 2u);
	ASSERT_EQ(place->delta.captures.size(), 2u);
	EXPECT_EQ(place->delta.captures[0].x, 3u);
	EXPECT_EQ(place->delta.captures[0].y, 4u);
	EXPECT_EQ(place->delta.captures[1].x, 5u);
	EXPECT_EQ(place->delta.captures[1].y, 6u);

	const auto pass = parseAs<network::ServerGameDelta>(R"({"type":"delta","moveId":8,"action":"pass","player":"white","next":"black"})");
	ASSERT_TRUE(pass.has_value());
	EXPECT_EQ(pass->delta.action, GameAction::Pass);
	EXPECT_EQ(pass->delta.player, Player::White);
	EXPECT_FALSE(pass->delta.coord.has_value());
	EXPECT_TRUE(pass->delta.captures.empty());

	const auto resigned = parseAs<network::ServerGameEnd>(R"({"type":"end","reason":"resignation","winner":"white"})");
	ASSERT_TRUE(resigned.has_value());
	EXPECT_EQ(resigned->result.reason, EndReason::Resignation);
	EXPECT_EQ(resigned->result.winner, Player::White);

	const auto counted = parseAs<network::ServerGameEnd>(R"({"type":"end","reason":"counting"})");
	ASSERT_TRUE(counted.has_value());
	EXPECT_EQ(counted->result.reason, EndReason::Counting);
	EXPECT_FALSE(counted->result.winner.has_value());

	const auto chat = parseAs<network::ServerChat>(R"({"type":"chat","player":"white","messageId":0,"message":"hello,world"})");
	ASSERT_TRUE(chat.has_value());
	EXPECT_EQ(chat->player, Player::White);
	EXPECT_EQ(chat->messageId, 0u);
	EXPECT_EQ(chat->message, "hello,world");
}

//! Newer peers may send more than we know. Fields a message does not need are ignored, not rejected.
TEST(GameNetMessages, ServerIgnoresUnneededFields) {
	const auto pass =
	        parseAs<network::ServerGameDelta>(R"({"type":"delta","moveId":8,"action":"pass","player":"white","next":"black","x":1,"y":2,"note":"?"})");
	ASSERT_TRUE(pass.has_value());
	EXPECT_EQ(pass->delta.action, GameAction::Pass);
	EXPECT_FALSE(pass->delta.coord.has_value());

	const auto start = parseAs<network::ServerGameStart>(
	        R"({"type":"start","boardSize":9,"time":300,"rules":{"scoring":"area","ko":"positional","komi":7.5,"suicide":true,"handicap":0}})");
	ASSERT_TRUE(start.has_value());
	EXPECT_EQ(start->config.boardSize, 9u);
	EXPECT_EQ(start->config.rules.koRule, Ko::Positional);
}

TEST(GameNetMessages, ServerFromMessageInvalid) {
	const auto rejects = [](const char* message) { EXPECT_FALSE(network::fromServerMessage(message).has_value()) << message; };

	rejects(R"({"type":"session"})");

	// Start: every rule is required, names must be known and the board size supported.
	rejects(R"({"type":"start","rules":{"scoring":"area","ko":"simple","komi":7.5,"suicide":true}})");
	rejects(R"({"type":"start","boardSize":7,"rules":{"scoring":"area","ko":"simple","komi":7.5,"suicide":true}})");
	rejects(R"({"type":"start","boardSize":9})");
	rejects(R"({"type":"start","boardSize":9,"rules":{"scoring":"bogus","ko":"simple","komi":7.5,"suicide":true}})");
	rejects(R"({"type":"start","boardSize":9,"rules":{"scoring":"area","komi":7.5,"suicide":true}})");
	rejects(R"({"type":"start","boardSize":9,"rules":{"scoring":"area","ko":"simple","komi":"bad","suicide":true}})");
	rejects(R"({"type":"start","boardSize":9,"rules":{"scoring":"area","ko":"simple","komi":7.5}})");

	// Delta: required fields, known names, a coordinate for every placement.
	rejects(R"({"type":"delta","action":"pass","player":"white","next":"black"})");
	rejects(R"({"type":"delta","moveId":1,"action":"jump","player":"white","next":"black"})");
	rejects(R"({"type":"delta","moveId":1,"action":"pass","player":"red","next":"black"})");
	rejects(R"({"type":"delta","moveId":1,"action":"pass","player":"white"})");
	rejects(R"({"type":"delta","moveId":1,"action":"place","player":"black","next":"white"})");
	rejects(R"({"type":"delta","moveId":1,"action":"place","player":"black","next":"white","x":1,"y":"2"})");
	rejects(R"({"type":"delta","moveId":1,"action":"place","player":"black","next":"white","x":1,"y":2,"captures":"bad"})");
	rejects(R"({"type":"delta","moveId":1,"action":"place","player":"black","next":"white","x":1,"y":2,"captures":[[1]]})");
	rejects(R"({"type":"delta","moveId":1,"action":"place","player":"black","next":"white","x":1,"y":2,"captures":[[1,"a"]]})");
	rejects(R"({"type":"delta","moveId":1,"action":0,"player":"black","next":"white","x":1,"y":2})"); // Enums go by name, not number.

	// End: the reason is required, names must be known.
	rejects(R"({"type":"end","winner":"white"})");
	rejects(R"({"type":"end","reason":"bogus"})");
	rejects(R"({"type":"end","reason":"resignation","winner":"red"})");

	rejects(R"({"type":"chat","player":"grey","messageId":0,"message":"hi"})");
	rejects(R"({"type":"chat","player":2,"messageId":0,"message":"hi"})");
	rejects(R"({"type":"chat","player":"white","messageId":0})");

	rejects(R"({"type":"unknown"})");
	rejects("not-json");
}

//! Every enum value survives the wire. Catches a value added to the model but missing from the wire names.
TEST(GameNetMessages, RoundTripsEveryEnumValue) {
	for (const auto action: {GameAction::Place, GameAction::Pass, GameAction::Resign}) {
		const auto coord  = action == GameAction::Place ? std::optional{Coord{2u, 3u}} : std::nullopt;
		const auto parsed = roundTrip(network::ServerGameDelta{makeMove(1u, action, Player::Black, coord)});
		ASSERT_TRUE(parsed.has_value());
		EXPECT_EQ(parsed->delta.action, action);
	}

	for (const auto player: {Player::Black, Player::White}) {
		const auto parsed = roundTrip(network::ServerGameDelta{makeMove(1u, GameAction::Pass, player)});
		ASSERT_TRUE(parsed.has_value());
		EXPECT_EQ(parsed->delta.player, player);
		EXPECT_EQ(parsed->delta.nextPlayer, opponent(player));
	}

	for (const auto reason: {EndReason::Resignation, EndReason::Counting, EndReason::Timeout, EndReason::Forfeit}) {
		const auto parsed = roundTrip(network::ServerGameEnd{GameResult{.winner = Player::Black, .reason = reason}});
		ASSERT_TRUE(parsed.has_value());
		EXPECT_EQ(parsed->result.reason, reason);
		EXPECT_EQ(parsed->result.winner, Player::Black);
	}

	for (const auto scoring: {Scoring::Territory, Scoring::Area}) {
		for (const auto ko: {Ko::Simple, Ko::Situational, Ko::Positional}) {
			const GameConfig config{.boardSize = 9u, .rules = {.scoringMethod = scoring, .koRule = ko, .komi = 6.5f, .suicideLegal = false}};
			const auto parsed = roundTrip(network::ServerGameStart{config});
			ASSERT_TRUE(parsed.has_value());
			EXPECT_EQ(parsed->config.rules.scoringMethod, scoring);
			EXPECT_EQ(parsed->config.rules.koRule, ko);
		}
	}

	for (const auto size: SUPPORTED_BOARD_SIZES) {
		const auto parsed = roundTrip(network::ServerGameStart{GameConfig{.boardSize = size, .rules = fromRuleSet(RuleSet::Japanese)}});
		ASSERT_TRUE(parsed.has_value()) << "Board size " << size;
		EXPECT_EQ(parsed->config.boardSize, size);
	}
}

TEST(GameNetMessages, ServerOmitsEmptyFields) {
	using nlohmann::json;

	const auto pass = json::parse(network::toMessage(network::ServerGameDelta{makeMove(9u, GameAction::Pass, Player::Black)}));
	EXPECT_FALSE(pass.contains("x"));
	EXPECT_FALSE(pass.contains("y"));
	EXPECT_FALSE(pass.contains("captures"));

	const auto counted = json::parse(network::toMessage(network::ServerGameEnd{GameResult{.winner = std::nullopt, .reason = EndReason::Counting}}));
	EXPECT_FALSE(counted.contains("winner"));
}

// NDEBUG disables the assert() this death test relies on, so it can only run in debug builds.
#if GTEST_HAS_DEATH_TEST && !defined(NDEBUG)
TEST(GameNetMessages, ServerDeltaMissingXYSerialization) {
	const auto build = [] { network::toMessage(network::ServerGameDelta{makeMove(1u, GameAction::Place, Player::Black)}); };
	EXPECT_DEATH(build(), "coord");
}
#endif

} // namespace tengen::gtest
