#pragma once

#include "model/player.hpp"

#include <filesystem>

namespace tengen::engine {

//! Where KataGo and its networks live. The engine catalog fills it in.
struct KataGoFiles {
	std::filesystem::path executable; //!< Executable path.
	std::filesystem::path model;      //!< KataGo's own network.
	std::filesystem::path humanModel; //!< The network trained on human games. It is what imitates a rank.
	std::filesystem::path gtpConfig;  //!< KataGo's GTP configuration, one made for the human model.
};

//! Where KataGo is installed and how it plays.
struct KataGoConfig {
	static constexpr Skill weakestRank   = fromKyu(20); //!< Weakest skill our model supports.
	static constexpr Skill strongestRank = fromDan(3);  //!< Highest rank our model supports.

	KataGoFiles files;       //!< Where KataGo and its networks live.
	Skill rank{weakestRank}; //!< Which level the engine should play at.
};

} // namespace tengen::engine
