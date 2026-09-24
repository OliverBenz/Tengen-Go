#include "engine/engineCatalog.hpp"

#include "engine/gnuGo.hpp"
#include "engine/kataGo.hpp"

#include <algorithm>
#include <initializer_list>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

namespace tengen::engine {

// The one place that knows where the engines live, each in its own directory below the engine root:
//
//   <engine root>/
//     gnugo/
//       gnugo(.exe)
//     katago/
//       katago(.exe)
//       model.bin.gz
//       human_model.bin.gz
//       gtp.cfg
//
// Keep the README's Bot Games section in line with it.

//! The name an executable has on this platform.
static std::string executableFile(const std::string& name) {
#ifdef _WIN32
	return name + ".exe";
#else
	return name;
#endif
}

static GnuGoFiles gnuGoFiles(const std::filesystem::path& engineRoot) {
	const std::filesystem::path directory = engineRoot / "gnugo";
	return {.executable = directory / executableFile("gnugo")};
}

// TODO: The file names are placeholders until the bot dialog offers KataGo.
static KataGoFiles kataGoFiles(const std::filesystem::path& engineRoot) {
	const std::filesystem::path directory = engineRoot / "katago";
	return {.executable = directory / executableFile("katago"),
	        .model      = directory / "model.bin.gz",
	        .humanModel = directory / "human_model.bin.gz",
	        .gtpConfig  = directory / "gtp.cfg"};
}

//! Whether every file is there. A file we cannot even look at counts as missing.
static bool allExist(const std::initializer_list<std::filesystem::path> files) {
	std::error_code ec;
	return std::ranges::all_of(files, [&ec](const std::filesystem::path& file) { return std::filesystem::exists(file, ec); });
}

static bool installed(const GnuGoFiles& files) {
	return allExist({files.executable});
}

static bool installed(const KataGoFiles& files) {
	return allExist({files.executable, files.model, files.humanModel, files.gtpConfig});
}

//! GNU Go's config, as long as all of its files are there.
static std::optional<GnuGoConfig> findGnuGo(const std::filesystem::path& engineRoot) {
	const GnuGoFiles files = gnuGoFiles(engineRoot);
	if (!installed(files)) {
		return std::nullopt;
	}
	return GnuGoConfig{.files = std::move(files)};
}

//! KataGo's config, as long as all of its files are there.
static std::optional<KataGoConfig> findKataGo(const std::filesystem::path& engineRoot) {
	const KataGoFiles files = kataGoFiles(engineRoot);
	if (!installed(files)) {
		return std::nullopt;
	}
	return KataGoConfig{.files = std::move(files)};
}

InstalledEngines findEngines(const std::filesystem::path& engineRoot) {
	return {.gnuGo  = findGnuGo(engineRoot),
	        .kataGo = findKataGo(engineRoot)};
}


static std::unique_ptr<GtpEngine> make(const GnuGoConfig& config) {
	return std::make_unique<GnuGo>(config);
}
static std::unique_ptr<GtpEngine> make(const KataGoConfig& config) {
	return std::make_unique<KataGo>(config);
}
std::unique_ptr<GtpEngine> makeEngine(const EngineConfig& config) {
	// One make() per engine config: a config without one does not compile.
	return std::visit([](const auto& engineConfig) { return make(engineConfig); }, config);
}

} // namespace tengen::engine
