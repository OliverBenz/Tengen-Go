#pragma once

#include "tengen/IAppSignalListener.hpp"
#include "model/player.hpp"

#include <vector>
#include <string>

namespace tengen::app {

struct ChatEntry {
	Player player;
	unsigned messageId;
	std::string message;
};

class IChatSession {
public:
    virtual ~IChatSession() = default;

    virtual void subscribe(app::IAppSignalListener* listener, uint64_t mask) = 0;
    virtual void unsubscribe(app::IAppSignalListener* listener) = 0;

    virtual void chat(const std::string& message) = 0;
    virtual std::vector<ChatEntry> getChatSince(unsigned messageId) const = 0;
};

}
