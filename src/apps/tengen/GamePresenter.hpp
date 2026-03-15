#pragma once

#include "BoardPresenter.hpp"
#include "ChatPresenter.hpp"
#include "tengen/sessionManager.hpp"

#include <memory>

namespace tengen {

namespace gui {
class GameWidget;
} // namespace gui

class GamePresenter : public app::IAppSignalListener {
public:
	GamePresenter(app::SessionManager& game, gui::GameWidget& gameWidget);
	~GamePresenter() override;

	//! Called by the game thread. Ensure not blocking.
	void onAppEvent(app::AppSignal signal) override;

private:
	app::SessionManager& m_game;
	gui::GameWidget& m_gameWidget;
	std::unique_ptr<BoardPresenter> m_boardPresenter = nullptr;
	std::unique_ptr<ChatPresenter> m_chatPresenter   = nullptr;
};

} // namespace tengen
