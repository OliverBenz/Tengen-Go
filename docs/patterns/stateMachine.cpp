// State machine: strongly typed, checked at compile time, no runtime overhead.
// Depending on the current state, an event moves the machine into a new state.
//
// Built from states and events:
//  - State: what the machine is currently in.
//  - Event: something that happens. Each state may handle it differently.
//
// Example below: a simplified network server.
//  - States: Idle, Running, Closed
//  - Events: Start, Connect, Disconnect, Stop
//
#include <cassert>
#include <cstdint>
#include <variant>

// States as a variant of structs.
struct Idle {};
struct Running {
	std::uint16_t port;
};
struct Closed {};
using State = std::variant<Idle, Running, Closed>;


// Events as a variant of structs.
struct Start {
	std::uint16_t port;
};
struct Connect {
	std::uint32_t connectionId;
};
struct Disconnect {
	std::uint32_t connectionId;
};
struct Stop {};
using Event = std::variant<Start, Connect, Disconnect, Stop>;


class Server {
public:
	constexpr void on_event(const Event& event) {
		// The handler of the current state returns the next one. Handlers are the private overloads below.
		m_state = std::visit(
		        [&](const auto& state) -> State { return on_event(state, event); },
		        m_state);
	}

	constexpr const State& state() const {
		return m_state;
	}

private:
	constexpr State on_event(const Idle& state, const Event& event) {
		if (const auto* start = std::get_if<Start>(&event))
			return Running{start->port};

		return Idle{};
	}

	constexpr State on_event(const Running& state, const Event& event) {
		if (std::holds_alternative<Stop>(event))
			return Closed{};

		return state;
	}

	constexpr State on_event(const Closed& state, const Event& event) {
		return state;
	}

private:
	State m_state{Idle{}};
};


// Here, we just show that state transitions are verified at compile time
constexpr bool transitions_correctly() {
	Server server;
	server.on_event(Start{8080});
	if (!std::holds_alternative<Running>(server.state()))
		return false;

	server.on_event(Stop{});
	return std::holds_alternative<Closed>(server.state());
}

static_assert(transitions_correctly());

int main() {
}
