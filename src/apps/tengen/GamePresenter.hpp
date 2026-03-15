#pragma once

#include "BoardPresenter.hpp"
#include "ChatPresenter.hpp"
#include "gui/gameWidget.hpp"
#include "tengen/sessionManager.hpp"

#include <memory>

namespace tengen {

class GamePresenter : public app::IAppSignalListener {
public:
	GamePresenter(app::SessionManager& game, gui::GameWidget& gameWidget);
	~GamePresenter() override;

	void onAppEvent(app::AppSignal signal) override; //!< Called by the game thread. Ensure not blocking.
	void onPassRequested();                          //!< Handle pass event from Board Widget.
	void onResignRequested();                        //!< Handle resign event from Board Widget.

private:
	app::SessionManager& m_game;
	gui::GameWidget& m_gameWidget;
	std::unique_ptr<BoardPresenter> m_boardPresenter = nullptr;
	std::unique_ptr<ChatPresenter> m_chatPresenter   = nullptr;
};

} // namespace tengen
