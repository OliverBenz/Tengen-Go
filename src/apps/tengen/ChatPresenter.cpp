#include "ChatPresenter.hpp"

#include "gui/chatWidget.hpp"

#include <QMetaObject>
#include <QObject>

#include <format>
#include <vector>

namespace tengen {

ChatPresenter::ChatPresenter(app::IChatSession& chat, gui::ChatWidget& chatWidget) : m_chat(chat), m_chatWidget(chatWidget) {
	QObject::connect(&m_chatWidget, &gui::ChatWidget::chatEvent, this, &ChatPresenter::onChatRequested);

	m_chat.subscribe(this, app::AS_NewChat);
	onAppEvent(app::AS_NewChat);
}

ChatPresenter::~ChatPresenter() {
	m_chat.unsubscribe(this);
}

void ChatPresenter::onChatRequested(const std::string& message) {
	m_chat.chat(message);
}

void ChatPresenter::onAppEvent(const app::AppSignalMask signal) {
	if ((signal & app::AS_NewChat) == 0) {
		return;
	}

	// Get new messages
	const auto messageEntries = m_chat.getChatSince(m_lastChatMessageId);
	if (messageEntries.empty()) {
		return;
	}

	// Construct new chat messages
	std::vector<std::string> lines;
	lines.reserve(messageEntries.size());
	for (const auto& entry: messageEntries) {
		const auto player = entry.player == Player::Black ? "Black" : "White";
		lines.emplace_back(std::format("{}: {}", player, entry.message));
		m_lastChatMessageId = entry.messageId;
	}

	// Append in GUI
	for (const auto& line: lines) {
		m_chatWidget.appendMessage(line);
	}
}

} // namespace tengen
