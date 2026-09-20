#pragma once

#include "core/IGameStateListener.hpp"
#include "core/game.hpp"
#include "engine/kataGo.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/position.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

namespace tengen::app {

//! Play locally against a bot engine.
//! The engine is brought up and asked for its moves on its own thread, and its answers are pushed into
//! the Game like any other move. The Game stays the source of truth; the Position only follows once
//! the Game accepted it.
class BotSession : public IGameSession, public IGameStateListener {
public:
	enum class Status {
		Idle,     //!< The engine is not up yet. The board takes no moves.
		BotMove,  //!< Bot's turn. The move has not been requested yet.
		Thinking, //!< Move requested. The engine answers on its own thread.
		PlayerMove,
		Finished
	};

	//! The engine configuration carries the strength: it names the KataGo config the bot plays with.
	BotSession(unsigned boardSize, const engine::LaunchConfig& engineConfig, bool playerPlaysAsBlack);
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

private:
	void relayPlayerMove(const GameDelta& delta);  //!< Mirror a move the Game accepted into the engine.
	void requestBotMove();                         //!< Ask the engine for its move without blocking the game loop.
	void pushBotMove(const engine::BotMove& move); //!< Hand the engine's move to the Game for validation.
	void joinEngineThread();                       //!< Wait for the startup thread to finish.
	void endSession(const std::string& reason);    //!< The bot cannot answer anymore: log it and close the session.

private:
	// Game Specifics
	Game m_game;           //!< Game instance. Run locally on bot games.
	Position m_position{}; //!< Tracks the board state as signalled by the Game.
	EventHub m_eventHub;   //!< Event notifier.

	// Bot specifics
	std::atomic<Status> m_status{Status::Idle}; //!< Also written from the engine thread.
	std::atomic<bool> m_shuttingDown{false};    //!< Set before the engine is stopped. Tells an aborted request from a failure.
	// TODO: place() runs on the game thread while genmove() may still block on the engine thread.
	// Both read the same pipe, so the engine still needs a lock of its own.
	engine::KataGo m_engine;           //!< The engine process. Runs genmove() requests on its own thread.
	Player m_botColour{Player::White}; //!< Colour the bot plays. The user takes the other one.
	std::thread m_engineThread;        //!< Runs the initial start()+startGame() sequence. Retired by shutdown().

	std::thread m_gameThread;        //!< Runs the game loop.
	mutable std::mutex m_stateMutex; //!< Concurrency handling.
};

} // namespace tengen::app
