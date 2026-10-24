#pragma once

#include "model/chatMessage.hpp"

#include <unordered_map>
#include <vector>

namespace tengen::app {

//! The chat as a session knows it. Puts the messages back in the order of their ids.
class SessionChatInfo {
public:
	void reset();                                                //!< Forget all messages.
	bool update(const ChatMessage& chat);                        //!< True if the history grew. Holds back messages that came early.
	std::vector<ChatMessage> getSince(unsigned messageId) const; //!< The messages after this id, in order.

private:
	unsigned m_expectedMessageId{1u};                    //!< Next expected chat message id.
	std::vector<ChatMessage> m_history;                  //!< Chat history.
	std::unordered_map<unsigned, ChatMessage> m_pending; //!< Messages received out of order.
};

} // namespace tengen::app
