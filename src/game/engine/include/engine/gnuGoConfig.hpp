#pragma once

#include <filesystem>

namespace tengen::engine {

//! Where GNU Go is installed. The engine catalog fills it in.
struct GnuGoFiles {
	std::filesystem::path executable;
};

//! Where GNU Go is installed and how it plays.
struct GnuGoConfig {
	// GnuGo uses levels 1-10 instead of a kyu/dan based ranking system.
	static constexpr int weakestLevel   = 1;
	static constexpr int strongestLevel = 10;

	GnuGoFiles files;        //!< Where GNU Go is installed.
	int level{weakestLevel}; //!< Which level the engine should play at.
};

} // namespace tengen::engine
