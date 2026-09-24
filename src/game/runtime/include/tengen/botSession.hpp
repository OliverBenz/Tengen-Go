#pragma once

#include "core/IGameStateListener.hpp"
#include "core/game.hpp"
#include "engine/gtpEngine.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/position.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace tengen::app {

//! Play locally against a bot engine.
//! The engine answers on a thread of its own and signals us through the listener interface. Its moves are
//! pushed into the Game like any other move. The Game stays the source of truth; the Position only
//! follows once the Game accepted it.
class BotSession : public IGameSession, public IGameStateListener, public engine::IEngineListener {
public:
	enum class Status {
		Idle,     //!< The engine is not up yet. The board takes no moves.
		BotMove,  //!< Bot's turn. The move has not been requested yet.
		Thinking, //!< Move requested. The engine answers on its own thread.
		PlayerMove,
		Finished
	};

	//! The session takes the engine over: it starts it for this game and shuts it down with it.
	//! How the bot plays is the engine's own config, so the session never needs to know it.
	BotSession(unsigned boardSize, std::unique_ptr<engine::GtpEngine> botEngine, bool playerPlaysAsBlack);
	~BotSession() override;

public: // IGameSession Interface
	GameStatus status() const override;
	Board board() const override;
	Player currentPlayer() const override;

	void tryPlace(unsigned x, unsigned y) override;
	void tryPass() override;
	void tryResign() override;
	void shutdown() override;

public: // IAppSignalSource Interface
	void subscribe(app::IAppSignalListener* listener, uint64_t mask) override;
	void unsubscribe(app::IAppSignalListener* listener) override;

public: // IGameStateListener Interface
	void onGameDelta(const GameDelta& delta) override;

public: // IEngineListener Interface
	void onEngineReady() override;
	void onMoveGenerated(const engine::BotMove& move) override;
	void onEngineFailed() override;

private:
	void relayPlayerMove(const GameDelta& delta); //!< Mirror a move the Game accepted into the engine.
	void requestBotMove();                        //!< Ask the engine for its move without blocking the game loop.
	void endSession(const std::string& reason);   //!< The bot cannot answer anymore: log it and close the session.

private:
	// Game Specifics
	Game m_game;           //!< Game instance. Run locally on bot games.
	Position m_position{}; //!< Tracks the board state as signalled by the Game.
	EventHub m_eventHub;   //!< Event notifier.

	// Bot specifics
	std::atomic<Status> m_status{Status::Idle};  //!< Also written from the engine thread.
	std::unique_ptr<engine::GtpEngine> m_engine; //!< The engine process. Runs its long requests on its own thread.
	Player m_botColour{Player::White};           //!< Colour the bot plays. The user takes the other one.

	std::thread m_gameThread;        //!< Runs the game loop.
	mutable std::mutex m_stateMutex; //!< Concurrency handling.
};

} // namespace tengen::app
