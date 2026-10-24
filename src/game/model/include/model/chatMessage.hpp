#pragma once

#include "model/player.hpp"

#include <string>

namespace tengen {

//! One thing a player said during the game.
struct ChatMessage {
	unsigned messageId; //!< Order in the game's chat, starting at 1.
	Player player;      //!< Who said it.
	std::string text;   //!< What was said.
};

} // namespace tengen
