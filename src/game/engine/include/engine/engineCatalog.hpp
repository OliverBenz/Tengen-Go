#pragma once

#include "engine/gnuGoConfig.hpp"
#include "engine/kataGoConfig.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <variant>

namespace tengen::engine {

class GtpEngine;

//! Configuration where the engine is installed and how it should play.
using EngineConfig = std::variant<GnuGoConfig, KataGoConfig>;

//! Which engines are installed below the engine root.
//! \note An engine is only set when all of its files were found. Its config then points at them.
struct InstalledEngines {
	std::optional<GnuGoConfig> gnuGo{std::nullopt};   //!< Set when GnuGo is installed.
	std::optional<KataGoConfig> kataGo{std::nullopt}; //!< Set when KataGo is installed.
};

//! Look for every engine in its own directory below the engineRoot.
InstalledEngines findEngines(const std::filesystem::path& engineRoot);

//! Create an engine based on the given configuration.
//! \note Does not check the installation: an engine missing its files fails to start and answers with onEngineFailed().
std::unique_ptr<GtpEngine> makeEngine(const EngineConfig& config);

} // namespace tengen::engine
