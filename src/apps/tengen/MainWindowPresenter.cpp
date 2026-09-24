#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "Logging.hpp"
#include "engine/engineCatalog.hpp"
#include "tengen/botSession.hpp"
#include "tengen/networkSession.hpp"
#include "tengen/openSession.hpp"

#include <QCoreApplication>
#include <QObject>
#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>

namespace tengen {
namespace {

//! The engines live in engine/ next to our executable. The engine catalog knows the layout below it.
std::filesystem::path engineRoot() {
	const std::filesystem::path appDir = QCoreApplication::applicationDirPath().toStdWString();
	return appDir / "engine";
}

//! The bot dialog still offers ranks, while GNU Go plays at a level.
//! TODO: Remove once the bot dialog offers GNU Go's levels itself.
int gnuGoLevel(const Skill botSkill) {
	// GNU Go's levels are not calibrated to ranks. Until someone plays them against known ranks, they
	// spread evenly from 20k to 6k, and any stronger skill gets GNU Go's strongest level.
	constexpr Skill weakest   = fromKyu(20);
	constexpr Skill strongest = fromKyu(6);
	constexpr int levels      = engine::GnuGoConfig::strongestLevel - engine::GnuGoConfig::weakestLevel;

	const int ranksAbove = stoneGap(weakest, std::clamp(botSkill, weakest, strongest));
	return engine::GnuGoConfig::weakestLevel + ranksAbove * levels / stoneGap(weakest, strongest);
}

} // namespace

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow)
    : QObject(nullptr), m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameBotRequested, this, &MainWindowPresenter::onNewBotGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, this, &MainWindowPresenter::onConnectRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, this, &MainWindowPresenter::onHostRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, this, &MainWindowPresenter::onShutdownRequested);

	// Bot games are only offered when their engine is installed.
	m_mainWindow.setBotGameAvailable(engine::findEngines(engineRoot()).gnuGo.has_value());

	startOpenPlay();
}

MainWindowPresenter::~MainWindowPresenter() = default;

void MainWindowPresenter::startOpenPlay() {
	m_gameSession   = std::make_unique<app::OpenSession>(9u);
	m_gamePresenter = std::make_unique<GamePresenter>(*m_gameSession, m_mainWindow.gameWidget());
}

void MainWindowPresenter::onNewLocalGameRequested() {
	onShutdownRequested();
	startOpenPlay();
}

void MainWindowPresenter::onNewBotGameRequested(unsigned boardSize, Skill botSkill, bool humanPlaysBlack) {
	// Look again rather than trust the menu: GNU Go may be gone since it was offered. Then the current game stays.
	std::optional<engine::GnuGoConfig> gnuGo = engine::findEngines(engineRoot()).gnuGo;
	if (!gnuGo) {
		gui::Logger().Log(Logging::LogLevel::Error, "GNU Go is not installed anymore. The bot game was not started.");
		return;
	}
	gnuGo->level = gnuGoLevel(botSkill);

	onShutdownRequested();

	m_gameSession   = std::make_unique<app::BotSession>(boardSize, engine::makeEngine(*gnuGo), humanPlaysBlack);
	m_gamePresenter = std::make_unique<GamePresenter>(*m_gameSession, m_mainWindow.gameWidget());
}

void MainWindowPresenter::onConnectRequested(const QString& hostIp) {
	onShutdownRequested();

	auto session = std::make_unique<app::NetworkSession>();
	session->connect(hostIp.toStdString());

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_gameSession = std::move(session);
}

void MainWindowPresenter::onHostRequested(const unsigned boardSize) {
	onShutdownRequested();

	auto session = std::make_unique<app::NetworkSession>();
	session->host(boardSize);

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_gameSession = std::move(session);
}

void MainWindowPresenter::onShutdownRequested() {
	if (m_gameSession) {
		m_gamePresenter = nullptr; // Destroy before m_game
		m_gameSession->shutdown();
		m_gameSession.reset();
	}
}

} // namespace tengen
