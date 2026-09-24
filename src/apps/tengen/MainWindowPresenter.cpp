#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "engine/engineCatalog.hpp"
#include "tengen/botSession.hpp"
#include "tengen/networkSession.hpp"
#include "tengen/openSession.hpp"

#include <QCoreApplication>
#include <QObject>
#include <filesystem>
#include <memory>

namespace tengen {

//! The engines live in engine/ next to our executable. The engine catalog knows the layout below it.
static std::filesystem::path engineRoot() {
	const std::filesystem::path appDir = QCoreApplication::applicationDirPath().toStdWString();
	return appDir / "engine";
}


MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow)
    : QObject(nullptr), m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::botDialogRequested, this, &MainWindowPresenter::onBotDialogRequested);
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

void MainWindowPresenter::onBotDialogRequested() {
	// Look on every opening, so an engine installed while we run is offered right away.
	m_mainWindow.openBotDialog(engine::findEngines(engineRoot()));
}

void MainWindowPresenter::onNewBotGameRequested(unsigned boardSize, const engine::EngineConfig& engineConfig, bool humanPlaysBlack) {
	onShutdownRequested();

	m_gameSession   = std::make_unique<app::BotSession>(boardSize, engine::makeEngine(engineConfig), humanPlaysBlack);
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
