#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "tengen/botSession.hpp"
#include "tengen/networkSession.hpp"
#include "tengen/openSession.hpp"

#include <QObject>
#include <filesystem>
#include <memory>

namespace tengen {
namespace {

//! Engine assets of the local development setup.
//! TODO: Ship the engine with the build instead of reading it out of the source tree.
engine::LaunchConfig localEngineConfig() {
	const std::filesystem::path configDir = TENGEN_CONFIG_DIR;
	const std::filesystem::path binDir    = configDir / "bin";

	return engine::LaunchConfig{
	        .executable = (binDir / "katago").string(),
	        .model      = (binDir / "g170-b30c320x2-s4824661760-d1229536699.bin.gz").string(),
	        .modelHuman = (binDir / "b18c384nbt-humanv0.bin.gz").string(),
	        .config     = (configDir / "gtp_human5k_example.cfg").string()};
}

} // namespace

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow)
    : QObject(nullptr), m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameBotRequested, this, &MainWindowPresenter::onNewBotGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, this, &MainWindowPresenter::onConnectRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, this, &MainWindowPresenter::onHostRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, this, &MainWindowPresenter::onShutdownRequested);

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

void MainWindowPresenter::onNewBotGameRequested(unsigned boardSize, gui::Difficulty difficulty, bool humanPlaysBlack) {
	onShutdownRequested();

	// TODO: Pick the engine strength. The local setup only holds the 5k configuration.
	(void)difficulty;

	m_gameSession   = std::make_unique<app::BotSession>(boardSize, localEngineConfig(), humanPlaysBlack);
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
