#include "model/gameRules.hpp"
#include "tengen/gameServer.hpp"

#include <iostream>

int main(int, char**) {
	tengen::app::GameServer server(9u, tengen::fromRuleSet(tengen::RuleSet::Japanese), tengen::Player::Black);
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
