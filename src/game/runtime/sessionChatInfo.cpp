#include "tengen/sessionChatInfo.hpp"

#include <algorithm>

namespace tengen::app {

void SessionChatInfo::reset() {
	m_expectedMessageId = 1u;
	m_history.clear();
	m_pending.clear();
}

bool SessionChatInfo::update(const ChatEntry& entry) {
	bool appended = false;

	if (entry.messageId < m_expectedMessageId) {
		// Ignore already seen messages.
	} else if (entry.messageId == m_expectedMessageId) {
		m_history.emplace_back(entry);
		++m_expectedMessageId;
		appended = true;
	} else {
		m_pending.emplace(entry.messageId, entry);
	}

	// Try inserting pending chat messages to history.
	while (true) {
		auto it = m_pending.find(m_expectedMessageId);
		if (it == m_pending.end()) {
			break;
		}
		m_history.emplace_back(it->second);
		m_pending.erase(it);
		++m_expectedMessageId;
		appended = true;
	}
	return appended;
}

std::vector<ChatEntry> SessionChatInfo::getSince(const unsigned messageId) const {
	const auto it = std::ranges::upper_bound(m_history, messageId, {}, &ChatEntry::messageId);
	return {it, m_history.end()};
}

} // namespace tengen::app
