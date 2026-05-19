#pragma once

#include "tengen/IChatSession.hpp"

namespace tengen {

namespace gui {
class ChatWidget;
}

class ChatPresenter : public app::IAppSignalListener {
public:
	ChatPresenter(app::IChatSession& chat, gui::ChatWidget& chatWidget);
	~ChatPresenter() override;

	void onAppEvent(app::AppSignal signal) override; //!< Called by the game thread. Ensure not blocking.

private:
	app::IChatSession& m_chat;
	gui::ChatWidget& m_chatWidget;

	unsigned m_lastChatMessageId = 0u;
};

} // namespace tengen
