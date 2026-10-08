#include "model/gameConfig.hpp"
#include "model/gameRules.hpp"
#include "tengen/gameServer.hpp"

#include <iostream>

int main(int, char**) {
	const tengen::GameConfig config{.boardSize = 9u, .rules = tengen::fromRuleSet(tengen::RuleSet::Japanese)};
	tengen::app::GameServer server(config, tengen::Player::Black);
	server.start();

	// NOTE: Can extend to allow more commands
	// Keep the server process alive until stdin closes or quit command.
	std::string line;
	while (std::getline(std::cin, line)) {
		if (line == "quit" || line == "exit") {
			break;
		}
	}

	server.stop();
	return 0;
}
