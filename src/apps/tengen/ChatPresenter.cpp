#include "ChatPresenter.hpp"

#include "ChatWidget.hpp"

#include <QMetaObject>
#include <QObject>

#include <format>
#include <vector>

namespace tengen {

ChatPresenter::ChatPresenter(app::SessionManager& game, gui::ChatWidget& chatWidget) : m_game(game), m_chatWidget(chatWidget) {
	QObject::connect(&m_chatWidget, &gui::ChatWidget::chatEvent, &m_chatWidget, [this](const std::string& message) { this->m_game->chat(message); });

	m_game.subscribe(this, app::AS_NewChat);
}

ChatPresenter::~ChatPresenter() {
	m_game.unsubscribe(this);
}

void ChatPresenter::onAppEvent(const app::AppSignal signal) {
	if (signal != app::AS_NewChat) {
		return;
	}

	const auto messageEntries = m_game.getChatSince(m_lastChatMessageId);
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
