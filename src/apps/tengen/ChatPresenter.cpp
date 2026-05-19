#include "ChatPresenter.hpp"

#include "gui/chatWidget.hpp"

#include <QMetaObject>
#include <QObject>

#include <format>
#include <vector>

namespace tengen {

ChatPresenter::ChatPresenter(app::IChatSession& chat, gui::ChatWidget& chatWidget) : m_chat(chat), m_chatWidget(chatWidget) {
	QObject::connect(&m_chatWidget, &gui::ChatWidget::chatEvent, &m_chatWidget, [this](const std::string& message) { m_chat.chat(message); });

	m_chat.subscribe(this, app::AS_NewChat);
	onAppEvent(app::AS_NewChat);
}

ChatPresenter::~ChatPresenter() {
	m_chat.unsubscribe(this);
}

void ChatPresenter::onAppEvent(const app::AppSignal signal) {
	if (signal != app::AS_NewChat) {
		return;
	}

	const auto messageEntries = m_chat.getChatSince(m_lastChatMessageId);
	if (messageEntries.empty()) {
		return;
	}

	std::vector<std::string> lines;
	lines.reserve(messageEntries.size());
	for (const auto& entry: messageEntries) {
		const auto player = entry.player == Player::Black ? "Black" : "White";
		lines.emplace_back(std::format("{}: {}", player, entry.message));
		m_lastChatMessageId = entry.messageId;
	}

	auto* widget = &m_chatWidget;
	QMetaObject::invokeMethod(
	        widget,
	        [widget, lines = std::move(lines)]() {
		        for (const auto& line: lines) {
			        widget->appendMessage(line);
		        }
	        },
	        Qt::QueuedConnection);
}

} // namespace tengen
