#include "ChatPresenter.hpp"

#include "gui/chatWidget.hpp"

#include <QMetaObject>
#include <QObject>

#include <format>
#include <string>

namespace tengen {

ChatPresenter::ChatPresenter(app::IChatSession& chat, gui::ChatWidget& chatWidget) : m_chat(chat), m_chatWidget(chatWidget) {
	QObject::connect(&m_chatWidget, &gui::ChatWidget::chatEvent, this, &ChatPresenter::onChatRequested);

	m_chat.subscribe(this, app::AS_NewChat); // Subscribe before the first read
	showNewMessages();
}

ChatPresenter::~ChatPresenter() {
	m_chat.unsubscribe(this);
}

void ChatPresenter::onChatRequested(const std::string& message) {
	m_chat.chat(message);
}

void ChatPresenter::onAppEvent(const app::AppSignal signal) {
	if (signal == app::AS_NewChat) {
		QMetaObject::invokeMethod(this, [this]() { showNewMessages(); }, Qt::QueuedConnection);
	}
}

void ChatPresenter::showNewMessages() {
	for (const auto& chat: m_chat.getChatSince(m_lastChatMessageId)) {
		m_chatWidget.appendMessage(std::format("{}: {}", toString(chat.player), chat.text));
		m_lastChatMessageId = chat.messageId;
	}
}

} // namespace tengen
