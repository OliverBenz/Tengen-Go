#pragma once

#include "model/chatMessage.hpp"
#include "tengen/IAppSignal.hpp"

#include <string>
#include <vector>

namespace tengen::app {

class IChatSession : public IAppSignalSource {
public:
	virtual ~IChatSession() = default;

	virtual void chat(const std::string& message)                           = 0;
	virtual std::vector<ChatMessage> getChatSince(unsigned messageId) const = 0;
};

} // namespace tengen::app
