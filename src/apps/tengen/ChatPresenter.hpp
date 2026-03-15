#pragma once

#include "tengen/sessionManager.hpp"

namespace tengen {

namespace gui {
class ChatWidget;
}

class ChatPresenter : public app::IAppSignalListener {
public:
	ChatPresenter(app::SessionManager& game, gui::ChatWidget& chatWidget);
	~ChatPresenter() override;

	void onAppEvent(app::AppSignal signal) override; //!< Called by the game thread. Ensure not blocking.

private:
	app::SessionManager& m_game;
	gui::ChatWidget& m_chatWidget;

	unsigned m_lastChatMessageId = 0u;
};

} // namespace tengen
