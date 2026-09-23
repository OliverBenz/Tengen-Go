#pragma once

#include "engine/gtpEngine.hpp"

#include <string>

namespace tengen::engine {

//! Where the engine and its assets live. This is deployment configuration and says nothing about how strong the bot plays: the strength comes in per game as a Skill.
struct LaunchConfig {
	std::string executable;
	std::string model;
	std::string modelHuman;
	std::string config;
};

//! Plays KataGo, imitating a human of the rank asked for.
//! \note The engine only imitates ranks it was trained on, so a skill outside that range plays at the closest one it has.
class KataGo : public GtpEngine {
public:
	explicit KataGo(LaunchConfig config);

	void start(unsigned boardSize, tengen::Player botColour, tengen::Skill botSkill) override;

private:
	LaunchConfig m_config;
};

} // namespace tengen::engine
