#pragma once

#include "model/player.hpp"
#include "tengen/IAppSignal.hpp"

#include <string>
#include <vector>

namespace tengen::app {

struct ChatEntry {
	Player player;
	unsigned messageId;
	std::string message;
};

class IChatSession : public IAppSignalSource {
public:
	virtual ~IChatSession() = default;

	virtual void chat(const std::string& message)                         = 0;
	virtual std::vector<ChatEntry> getChatSince(unsigned messageId) const = 0;
};

} // namespace tengen::app
