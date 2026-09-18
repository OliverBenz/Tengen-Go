#include "kataGo.hpp"

#include "model/coordinate.hpp"
#include "model/player.hpp"

#include <atomic>
#include <condition_variable>
#include <format>
#include <functional>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

constexpr const char* KATAGO_EXE         = KATAGO_PATH "/katago";
constexpr const char* KATAGO_MODEL       = KATAGO_PATH "/g170-b30c320x2-s4824661760-d1229536699.bin.gz";
constexpr const char* KATAGO_MODEL_HUMAN = KATAGO_PATH "/b18c384nbt-humanv0.bin.gz";
constexpr const char* KATAGO_CONFIG      = KATAGO_PATH "/gtp_human5k_example.cfg";

constexpr unsigned BOARD_SIZE = 19u;

//! Render a generated move for the console.
static std::string toString(const BotMove& move) {
	switch (move.action) {
	case MoveAction::Place:
		return std::format("({}, {})", move.coord.x, move.coord.y);
	case MoveAction::Pass:
		return "pass";
	case MoveAction::Resign:
		return "resign";
	}
	return "unknown";
}


class BotSession {
	enum class Status {
		Idle,
		// Active, // Active is implied by !Idle && !Finished
		BotMove,  //!< Bot's turn. The move has not been requested yet.
		Thinking, //!< Move requested. The engine answers through onBotMove().
		PlayerMove,
		Finished
	};

public:
	//! Notified once the bot picked a move. Runs on the engine thread.
	using MoveCallback = std::function<void(const BotMove& move)>;

	~BotSession() {
		joinEngineThread(); // The engine has to outlive the request that is using it.
	}

	void setMoveCallback(MoveCallback callback) {
		m_moveCallback = std::move(callback);
	}

	bool start(const unsigned boardSize) {
		if (m_status != Status::Idle && m_status != Status::Finished) {
			return false;
		}

		if (!m_engine.start(LaunchConfig{
		            .executable = KATAGO_EXE,
		            .model      = KATAGO_MODEL,
		            .modelHuman = KATAGO_MODEL_HUMAN,
		            .config     = KATAGO_CONFIG})) {
			return false;
		}

		// Player plays black and moves first.
		if (!m_engine.startGame(boardSize, tengen::Player::White)) {
			return false;
		}

		m_status = Status::PlayerMove;
		return true;
	}

	//! Register the player move. The move is already validated by the game rules.
	bool registerPlayerMove(const tengen::Coord position) {
		if (m_status != Status::PlayerMove) {
			return false;
		}

		if (!m_engine.place(position)) {
			return false;
		}

		m_status = Status::BotMove;
		return true;
	}

	//! Ask the engine for its move. Returns immediately while the bot thinks; onBotMove() delivers the result.
	bool requestBotMove() {
		if (m_status != Status::BotMove) {
			return false;
		}

		joinEngineThread(); // Retire the previous request. Only one is ever in flight.

		m_status       = Status::Thinking;
		m_engineThread = std::thread([this] {
			BotMove move{};
			const bool success = m_engine.genmove(move);
			onBotMove(success, move);
		});
		return true;
	}

	bool isActive() const {
		return m_status != Status::Idle && m_status != Status::Finished;
	}

private:
	//! Runs on the engine thread once the bot answered.
	void onBotMove(const bool success, const BotMove& move) {
		if (!success) {
			m_status = Status::Finished;
			return;
		}

		m_status = move.action == MoveAction::Resign ? Status::Finished : Status::PlayerMove;
		if (m_moveCallback) {
			m_moveCallback(move);
		}
	}

	//! Wait for the in flight request to finish.
	void joinEngineThread() {
		if (m_engineThread.joinable()) {
			m_engineThread.join();
		}
	}

private:
	// Bot specifics
	std::atomic<Status> m_status{Status::Idle}; //!< Also written from the engine thread.
	KataGo m_engine;                            //!< The engine process. Only touched by one thread at a time.
	std::thread m_engineThread;                 //!< Runs the in flight move request.

	// TODO: Replace this guy with the event hub.
	MoveCallback m_moveCallback; //!< Attached by the owner of the session.
};

int main() {
	std::cout << "Using configuration:\n"
	          << "  Executable:    " << KATAGO_EXE << '\n'
	          << "  Model:         " << KATAGO_MODEL << '\n'
	          << "  Human Model:   " << KATAGO_MODEL_HUMAN << '\n'
	          << "  Configuration: " << KATAGO_CONFIG << '\n';

	BotSession session;
	if (!session.start(BOARD_SIZE)) {
		std::cerr << "Failed to start the session\n";
		return 1;
	}

	std::mutex mutex;
	std::condition_variable condition;
	bool done = false;

	session.setMoveCallback([&](const BotMove& move) {
		std::cout << "Bot plays: " << toString(move) << '\n';
		{
			std::lock_guard<std::mutex> lock(mutex);
			done = true;
		}
		condition.notify_one();
	});

	// Our rows run from the top down, so Q4 on a 19x19 board is column 15, row 15.
	const tengen::Coord opening{15u, 15u};
	if (!session.registerPlayerMove(opening) || !session.requestBotMove()) {
		std::cerr << "Failed to play the move\n";
		return 1;
	}

	// The request returned while the bot is still thinking. A real owner (the game loop, the GUI)
	// keeps running here and reacts when the callback fires.
	std::cout << "Player played Q4, bot is thinking...\n";

	std::unique_lock<std::mutex> lock(mutex);
	condition.wait(lock, [&] { return done; });

	return 0;
}
