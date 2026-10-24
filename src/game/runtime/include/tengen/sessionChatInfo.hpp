#pragma once

#include "tengen/IChatSession.hpp"

#include <unordered_map>
#include <vector>

namespace tengen::app {

//! The chat as a session knows it. Puts the messages back in the order of their ids.
class SessionChatInfo {
public:
	void reset();                                              //!< Forget all messages.
	bool update(const ChatEntry& entry);                       //!< True if the history grew. Holds back entries that came early.
	std::vector<ChatEntry> getSince(unsigned messageId) const; //!< The messages after this id, in order.

private:
	unsigned m_expectedMessageId{1u};                  //!< Next expected chat message id.
	std::vector<ChatEntry> m_history;                  //!< Chat history.
	std::unordered_map<unsigned, ChatEntry> m_pending; //!< Messages received out of order.
};

} // namespace tengen::app
