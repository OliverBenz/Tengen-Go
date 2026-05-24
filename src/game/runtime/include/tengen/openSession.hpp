#pragma once

#include "core/IGameStateListener.hpp"
#include "core/game.hpp"
#include "tengen/IGameSession.hpp"
#include "tengen/eventHub.hpp"
#include "tengen/position.hpp"

#include <mutex>
#include <thread>

namespace tengen::app {

//! Free play locally you control both players.
class OpenSession : public IGameSession, public IGameStateListener {
public:
	OpenSession(std::size_t boardSize);
	~OpenSession() override;

	// GameSession Interface
	GameStatus status() const override;
	Board board() const override;
	Player currentPlayer() const override;

	void tryPlace(unsigned x, unsigned y) override;
	void tryPass() override;
	void tryResign() override;
	void shutdown() override;

	// AppSignal Handlers
	void subscribe(app::IAppSignalListener* listener, uint64_t mask) override;
	void unsubscribe(app::IAppSignalListener* listener) override;

	// Game Handlers
	void onGameDelta(const GameDelta& delta) override;

private:
	Game m_game;
	EventHub m_eventHub;

	std::thread m_gameThread; //!< Runs the game loop.
	mutable std::mutex m_stateMutex;

	Position m_position{};
};

} // namespace tengen::app
