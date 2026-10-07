#include "network/nwEvents.hpp"

#include <array>
#include <cassert>
#include <nlohmann/json.hpp>

namespace tengen::network {

using nlohmann::json;

//! Wire name of one enum value. Enums go over the wire by name - not by value.
template <typename E>
struct WireName {
	E value;
	const char* name;
};

// Never rename an entry: the names are the protocol.
constexpr std::array PLAYER_NAMES{
        WireName<Player>{Player::Black, "black"},
        WireName<Player>{Player::White, "white"},
};
constexpr std::array ACTION_NAMES{
        WireName<GameAction>{GameAction::Place, "place"},
        WireName<GameAction>{GameAction::Pass, "pass"},
        WireName<GameAction>{GameAction::Resign, "resign"},
};
constexpr std::array SCORING_NAMES{
        WireName<Scoring>{Scoring::Territory, "territory"},
        WireName<Scoring>{Scoring::Area, "area"},
};
constexpr std::array KO_NAMES{
        WireName<Ko>{Ko::Simple, "simple"},
        WireName<Ko>{Ko::Situational, "situational"},
        WireName<Ko>{Ko::Positional, "positional"},
};
constexpr std::array END_REASON_NAMES{
        WireName<EndReason>{EndReason::Resignation, "resignation"},
        WireName<EndReason>{EndReason::Counting, "counting"},
        WireName<EndReason>{EndReason::Timeout, "timeout"},
        WireName<EndReason>{EndReason::Forfeit, "forfeit"},
};

template <typename E, std::size_t N>
static const char* toName(const std::array<WireName<E>, N>& names, const E value) {
	for (const auto& entry: names) {
		if (entry.value == value) {
			return entry.name;
		}
	}
	assert(false && "Enum value missing from its wire names.");
	return "";
}

//! The enum value with this wire name. Nullopt for anything that is not one of the names.
template <typename E, std::size_t N>
static std::optional<E> fromName(const std::array<WireName<E>, N>& names, const json& j) {
	if (!j.is_string()) {
		return std::nullopt;
	}
	const auto& name = j.get_ref<const std::string&>();
	for (const auto& entry: names) {
		if (name == entry.name) {
			return entry.value;
		}
	}
	return std::nullopt;
}

static std::string toMessage(const ClientPutStone& e) {
	json j;
	j["type"] = "put";
	j["x"]    = e.c.x;
	j["y"]    = e.c.y;
	return j.dump();
}
static std::string toMessage(const ClientPass&) {
	json j;
	j["type"] = "pass";
	return j.dump();
}
static std::string toMessage(const ClientResign&) {
	json j;
	j["type"] = "resign";
	return j.dump();
}
static std::string toMessage(const ClientChat& e) {
	json j;
	j["type"]    = "chat";
	j["message"] = e.message;
	return j.dump();
}

std::string toMessage(ClientEvent event) {
	return std::visit([&](auto&& ev) { return toMessage(ev); }, event);
}

std::optional<ClientEvent> fromClientMessage(const std::string& message) {
	const auto j = json::parse(message, nullptr, false);
	if (!j.is_object()) {
		return {};
	}
	if (!j.contains("type") || !j["type"].is_string()) {
		return {};
	}
	const auto type = j["type"].get<std::string>();
	if (type == "put") {
		if (!j.contains("x") || !j.contains("y") || !j["x"].is_number_unsigned() || !j["y"].is_number_unsigned()) {
			return {};
		}
		return ClientPutStone{.c = {j["x"].get<unsigned>(), j["y"].get<unsigned>()}};
	}
	if (type == "pass") {
		return ClientPass{};
	}
	if (type == "resign") {
		return ClientResign{};
	}
	if (type == "chat") {
		if (!j.contains("message") || !j["message"].is_string()) {
			return {};
		}
		return ClientChat{.message = j["message"].get<std::string>()};
	}
	return {};
}

static std::string toMessage(const ServerSessionAssign& e) {
	json j;
	j["type"]      = "session";
	j["sessionId"] = e.sessionId;
	return j.dump();
}
static std::string toMessage(const ServerGameStart& e) {
	const auto& rules = e.config.rules;

	json j;
	j["type"]      = "start";
	j["boardSize"] = e.config.boardSize;
	j["rules"]     = {
            {"scoring", toName(SCORING_NAMES, rules.scoringMethod)},
            {"ko", toName(KO_NAMES, rules.koRule)},
            {"komi", rules.komi},
            {"suicide", rules.suicideLegal},
    };
	return j.dump();
}
static std::string toMessage(const ServerGameDelta& e) {
	const auto& delta = e.delta;

	json j;
	j["type"]   = "delta";
	j["moveId"] = delta.moveId;
	j["action"] = toName(ACTION_NAMES, delta.action);
	j["player"] = toName(PLAYER_NAMES, delta.player);
	j["next"]   = toName(PLAYER_NAMES, delta.nextPlayer);

	// For Place moves we require a coord; otherwise the message is invalid.
	if (delta.action == GameAction::Place) {
		if (!delta.coord.has_value()) {
			assert(false && "ServerGameDelta::Place requires coord");
			return {};
		}
		j["x"] = delta.coord->x;
		j["y"] = delta.coord->y;
		if (!delta.captures.empty()) {
			auto caps = json::array();
			for (const auto& cap: delta.captures) {
				caps.push_back({cap.x, cap.y});
			}
			j["captures"] = std::move(caps);
		}
	}

	return j.dump();
}
static std::string toMessage(const ServerGameEnd& e) {
	json j;
	j["type"]   = "end";
	j["reason"] = toName(END_REASON_NAMES, e.result.reason);
	if (e.result.winner) {
		j["winner"] = toName(PLAYER_NAMES, *e.result.winner);
	}
	return j.dump();
}

static std::string toMessage(const ServerChat& e) {
	json j;
	j["type"]      = "chat";
	j["player"]    = toName(PLAYER_NAMES, e.player);
	j["messageId"] = e.messageId;
	j["message"]   = e.message;
	return j.dump();
}
std::string toMessage(ServerEvent event) {
	return std::visit([&](auto&& ev) { return toMessage(ev); }, event);
}

static std::optional<ServerEvent> fromServerStartMessage(const json& j) {
	if (!j.contains("boardSize") || !j["boardSize"].is_number_unsigned() || !j.contains("rules") || !j["rules"].is_object()) {
		return {};
	}

	const auto& rules = j["rules"];
	if (!rules.contains("scoring") || !rules.contains("ko") || !rules.contains("komi") || !rules["komi"].is_number() || !rules.contains("suicide") || !rules["suicide"].is_boolean()) {
		return {};
	}
	const auto scoring = fromName(SCORING_NAMES, rules["scoring"]);
	const auto ko      = fromName(KO_NAMES, rules["ko"]);
	if (!scoring || !ko) {
		return {};
	}

	return ServerGameStart{GameConfig{
	        .boardSize = j["boardSize"].get<std::size_t>(),
	        .rules     = {.scoringMethod = *scoring, .koRule = *ko, .komi = rules["komi"].get<float>(), .suicideLegal = rules["suicide"].get<bool>()},
	}};
}

static std::optional<ServerEvent> fromServerDeltaMessage(const json& j) {
	if (!j.contains("moveId") || !j["moveId"].is_number_unsigned() || !j.contains("action") || !j.contains("player") || !j.contains("next")) {
		return {};
	}

	const auto action = fromName(ACTION_NAMES, j["action"]);
	const auto player = fromName(PLAYER_NAMES, j["player"]);
	const auto next   = fromName(PLAYER_NAMES, j["next"]);
	if (!action || !player || !next) {
		return {};
	}

	GameDelta delta{
	        .moveId     = j["moveId"].get<unsigned>(),
	        .action     = *action,
	        .player     = *player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = *next,
	};

	if (delta.action == GameAction::Place) {
		if (!j.contains("x") || !j.contains("y") || !j["x"].is_number_unsigned() || !j["y"].is_number_unsigned()) {
			return {};
		}
		delta.coord = Coord{.x = j["x"].get<unsigned>(), .y = j["y"].get<unsigned>()};
		if (j.contains("captures")) {
			if (!j["captures"].is_array()) {
				return {};
			}
			for (const auto& cap: j["captures"]) {
				if (!cap.is_array() || cap.size() != 2 || !cap[0].is_number_unsigned() || !cap[1].is_number_unsigned()) {
					return {};
				}
				delta.captures.push_back(Coord{.x = cap[0].get<unsigned>(), .y = cap[1].get<unsigned>()});
			}
		}
	} else {
		if (j.contains("x") || j.contains("y") || j.contains("captures")) {
			return {};
		}
	}

	return ServerGameDelta{delta};
}

static std::optional<ServerEvent> fromServerEndMessage(const json& j) {
	if (!j.contains("reason")) {
		return {};
	}
	const auto reason = fromName(END_REASON_NAMES, j["reason"]);
	if (!reason) {
		return {};
	}

	GameResult result{.winner = std::nullopt, .reason = *reason};
	if (j.contains("winner")) {
		const auto winner = fromName(PLAYER_NAMES, j["winner"]);
		if (!winner) {
			return {};
		}
		result.winner = *winner;
	}
	return ServerGameEnd{result};
}

std::optional<ServerEvent> fromServerMessage(const std::string& message) {
	const auto j = json::parse(message, nullptr, false);
	if (!j.is_object() || !j.contains("type") || !j["type"].is_string()) {
		return {};
	}
	const auto type = j["type"].get<std::string>();
	if (type == "session") {
		if (!j.contains("sessionId") || !j["sessionId"].is_number_unsigned()) {
			return {};
		}
		return ServerSessionAssign{.sessionId = j["sessionId"].get<SessionId>()};
	}
	if (type == "start") {
		return fromServerStartMessage(j);
	}
	if (type == "delta") {
		return fromServerDeltaMessage(j);
	}
	if (type == "end") {
		return fromServerEndMessage(j);
	}
	if (type == "chat") {
		if (!j.contains("player") || !j.contains("messageId") || !j["messageId"].is_number_unsigned() || !j.contains("message") || !j["message"].is_string()) {
			return {};
		}
		const auto player = fromName(PLAYER_NAMES, j["player"]);
		if (!player) {
			return {};
		}

		const auto chatMessageId = j["messageId"].get<unsigned>();
		const auto chatMessage   = j["message"].get<std::string>();
		return ServerChat{*player, chatMessageId, std::move(chatMessage)};
	}
	return {};
}

} // namespace tengen::network
