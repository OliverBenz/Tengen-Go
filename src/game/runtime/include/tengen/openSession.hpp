#pragma once

#include "core/IGameStateListener.hpp"
#include "core/game.hpp"
#include "model/gameConfig.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/sessionGameInfo.hpp"

#include <mutex>
#include <thread>

namespace tengen::app {

//! Free play locally you control both players.
class OpenSession : public IGameSession, public IGameStateListener {
public:
	//! \throws std::invalid_argument if the board size is not supported.
	explicit OpenSession(const GameConfig& config);
	~OpenSession() override;

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
	void onGameStart(const GameConfig& config) override;
	void onGameDelta(const GameDelta& delta) override;
	void onGameEnd(const GameResult& result) override;

private:
	Game m_game;                  //!< Game instance. Run locally on open sessions.
	SessionGameInfo m_gameInfo{}; //!< Tracks the game as signalled by the Game.
	EventHub m_eventHub;          //!< Event notifier.

	std::thread m_gameThread;        //!< Runs the game loop.
	mutable std::mutex m_stateMutex; //!< Concurrency handling.
};

} // namespace tengen::app
