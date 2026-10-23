#include "network/nwEvents.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
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
constexpr std::array SEAT_NAMES{
        WireName<Seat>{Seat::Black, "black"},
        WireName<Seat>{Seat::White, "white"},
        WireName<Seat>{Seat::Observer, "observer"},
};
constexpr std::array REJECT_REASON_NAMES{
        WireName<RejectReason>{RejectReason::ProtocolMismatch, "protocol"},
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
static std::optional<E> fromName(const std::array<WireName<E>, N>& names, const std::string& name) {
	for (const auto& entry: names) {
		if (name == entry.name) {
			return entry.value;
		}
	}
	return std::nullopt;
}

//! The value as T. Nullopt unless it is an unsigned number that fits: get<T>() alone cuts off the upper bits (2^32 + 1 becomes 1).
template <typename T>
static std::optional<T> readUnsigned(const json& value) {
	if (!value.is_number_unsigned()) {
		return std::nullopt;
	}
	const auto number = value.get<std::uint64_t>();
	if (number > std::numeric_limits<T>::max()) {
		return std::nullopt;
	}
	return static_cast<T>(number);
}

//! The field 'key' of j as T. Nullopt if it is missing, or not an unsigned number that fits.
template <typename T>
static std::optional<T> readUnsigned(const json& j, const char* key) {
	const auto it = j.find(key);
	if (it == j.end()) {
		return std::nullopt;
	}
	return readUnsigned<T>(*it);
}

static std::string toMessage(const ClientHello& e) {
	json j;
	j["type"]     = "hello";
	j["protocol"] = e.protocol;
	return j.dump();
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

std::string toMessage(const ClientEvent& event) {
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
	if (type == "hello") {
		const auto protocol = readUnsigned<ProtocolVersion>(j, "protocol");
		if (!protocol) {
			return {};
		}
		return ClientHello{.protocol = *protocol};
	}
	if (type == "put") {
		const auto x = readUnsigned<unsigned>(j, "x");
		const auto y = readUnsigned<unsigned>(j, "y");
		if (!x || !y) {
			return {};
		}
		return ClientPutStone{.c = {*x, *y}};
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

static std::string toMessage(const ServerWelcome& e) {
	json j;
	j["type"]      = "welcome";
	j["protocol"]  = e.protocol;
	j["sessionId"] = e.sessionId;
	j["seat"]      = toName(SEAT_NAMES, e.seat);
	return j.dump();
}
static std::string toMessage(const ServerReject& e) {
	json j;
	j["type"]   = "reject";
	j["reason"] = toName(REJECT_REASON_NAMES, e.reason);
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
std::string toMessage(const ServerEvent& event) {
	return std::visit([&](auto&& ev) { return toMessage(ev); }, event);
}

static std::optional<ServerEvent> fromServerWelcomeMessage(const json& j) {
	if (!j.contains("seat") || !j["seat"].is_string()) {
		return {};
	}
	const auto protocol  = readUnsigned<ProtocolVersion>(j, "protocol");
	const auto sessionId = readUnsigned<SessionId>(j, "sessionId");
	const auto seat      = fromName(SEAT_NAMES, j["seat"].get_ref<const std::string&>());
	if (!protocol || !sessionId || !seat) {
		return {};
	}
	return ServerWelcome{.protocol = *protocol, .sessionId = *sessionId, .seat = *seat};
}

static std::optional<ServerEvent> fromServerRejectMessage(const json& j) {
	if (!j.contains("reason") || !j["reason"].is_string()) {
		return {};
	}
	const auto reason = fromName(REJECT_REASON_NAMES, j["reason"].get_ref<const std::string&>());
	if (!reason) {
		return {};
	}
	return ServerReject{.reason = *reason};
}

static std::optional<ServerEvent> fromServerStartMessage(const json& j) {
	const auto boardSize = readUnsigned<std::size_t>(j, "boardSize");
	if (!boardSize || !j.contains("rules") || !j["rules"].is_object()) {
		return {};
	}

	const auto& rules = j["rules"];
	if (!rules.contains("scoring") || !rules["scoring"].is_string() || !rules.contains("ko") || !rules["ko"].is_string() || !rules.contains("komi") || !rules["komi"].is_number() || !rules.contains("suicide") ||
	    !rules["suicide"].is_boolean()) {
		return {};
	}
	const auto scoring = fromName(SCORING_NAMES, rules["scoring"].get_ref<const std::string&>());
	const auto ko      = fromName(KO_NAMES, rules["ko"].get_ref<const std::string&>());
	if (!scoring || !ko) {
		return {};
	}

	return ServerGameStart{GameConfig{
	        .boardSize = *boardSize,
	        .rules     = {.scoringMethod = *scoring, .koRule = *ko, .komi = rules["komi"].get<float>(), .suicideLegal = rules["suicide"].get<bool>()},
	}};
}

static std::optional<ServerEvent> fromServerDeltaMessage(const json& j) {
	if (!j.contains("action") || !j["action"].is_string() || !j.contains("player") || !j["player"].is_string() || !j.contains("next") || !j["next"].is_string()) {
		return {};
	}

	const auto moveId = readUnsigned<unsigned>(j, "moveId");
	const auto action = fromName(ACTION_NAMES, j["action"].get_ref<const std::string&>());
	const auto player = fromName(PLAYER_NAMES, j["player"].get_ref<const std::string&>());
	const auto next   = fromName(PLAYER_NAMES, j["next"].get_ref<const std::string&>());
	if (!moveId || !action || !player || !next) {
		return {};
	}

	GameDelta delta{
	        .moveId     = *moveId,
	        .action     = *action,
	        .player     = *player,
	        .coord      = std::nullopt,
	        .captures   = {},
	        .nextPlayer = *next,
	};

	if (delta.action == GameAction::Place) {
		const auto x = readUnsigned<unsigned>(j, "x");
		const auto y = readUnsigned<unsigned>(j, "y");
		if (!x || !y) {
			return {};
		}
		delta.coord = Coord{.x = *x, .y = *y};
		if (j.contains("captures")) {
			if (!j["captures"].is_array()) {
				return {};
			}
			for (const auto& cap: j["captures"]) {
				if (!cap.is_array() || cap.size() != 2) {
					return {};
				}
				const auto capX = readUnsigned<unsigned>(cap[0]);
				const auto capY = readUnsigned<unsigned>(cap[1]);
				if (!capX || !capY) {
					return {};
				}
				delta.captures.push_back(Coord{.x = *capX, .y = *capY});
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
	if (!j.contains("reason") || !j["reason"].is_string()) {
		return {};
	}
	const auto reason = fromName(END_REASON_NAMES, j["reason"].get_ref<const std::string&>());
	if (!reason) {
		return {};
	}

	GameResult result{.winner = std::nullopt, .reason = *reason};
	if (j.contains("winner")) {
		if (!j["winner"].is_string()) {
			return {};
		}
		const auto winner = fromName(PLAYER_NAMES, j["winner"].get_ref<const std::string&>());
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
	if (type == "welcome") {
		return fromServerWelcomeMessage(j);
	}
	if (type == "reject") {
		return fromServerRejectMessage(j);
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
		if (!j.contains("player") || !j["player"].is_string() || !j.contains("message") || !j["message"].is_string()) {
			return {};
		}
		const auto player    = fromName(PLAYER_NAMES, j["player"].get_ref<const std::string&>());
		const auto messageId = readUnsigned<unsigned>(j, "messageId");
		if (!player || !messageId) {
			return {};
		}

		const auto chatMessage = j["message"].get<std::string>();
		return ServerChat{*player, *messageId, std::move(chatMessage)};
	}
	return {};
}

} // namespace tengen::network
