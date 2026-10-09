#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "engine/engineCatalog.hpp"
#include "model/gameConfig.hpp"
#include "tengen/botSession.hpp"
#include "tengen/networkSession.hpp"
#include "tengen/openSession.hpp"

#include <QCoreApplication>
#include <QObject>
#include <QStandardPaths>
#include <filesystem>
#include <memory>
#include <vector>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow) : QObject(nullptr), m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameBotRequested, this, &MainWindowPresenter::onNewBotGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameHostRequested, this, &MainWindowPresenter::onHostRequested);

	QObject::connect(&m_mainWindow, &gui::MainWindow::botDialogRequested, this, &MainWindowPresenter::onBotDialogRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, this, &MainWindowPresenter::onConnectRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, this, &MainWindowPresenter::onShutdownRequested);

	startOpenPlay(GameConfig{.boardSize = 9u, .rules = fromRuleSet(RuleSet::Japanese)});
}

MainWindowPresenter::~MainWindowPresenter() = default;

void MainWindowPresenter::startOpenPlay(const GameConfig& config) {
	showLocalPlayers();
	m_gameSession   = std::make_unique<app::OpenSession>(config);
	m_gamePresenter = std::make_unique<GamePresenter>(*m_gameSession, m_mainWindow.gameWidget());
}

void MainWindowPresenter::showLocalPlayers() {
	m_mainWindow.gameWidget().setPlayers(tr("White"), tr("Black"), Player::White);
}

void MainWindowPresenter::showPlayers(const Player ownColour, const QString& opponentName) {
	m_mainWindow.gameWidget().setPlayers(tr("You"), opponentName, ownColour);
}

void MainWindowPresenter::onNewLocalGameRequested(const GameConfig& config) {
	onShutdownRequested();
	startOpenPlay(config);
}

void MainWindowPresenter::onBotDialogRequested() {
	// The user's data folder first (e.g. ~/.local/share/tengen/engine), next to our executable as the fallback.
	const std::vector<std::filesystem::path> rootPaths{
	        (QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/engine").toStdWString(),
	        (QCoreApplication::applicationDirPath() + "/engine").toStdWString(),
	};

	// Look on every opening, so an engine installed while we run is offered right away.
	m_mainWindow.openBotDialog(engine::findEngines(rootPaths));
}

void MainWindowPresenter::onNewBotGameRequested(const GameConfig& config, const engine::EngineConfig& engineConfig, const bool humanPlaysBlack) {
	onShutdownRequested();

	showPlayers(humanPlaysBlack ? Player::Black : Player::White, tr("Bot"));
	m_gameSession   = std::make_unique<app::BotSession>(config, engine::makeEngine(engineConfig), humanPlaysBlack);
	m_gamePresenter = std::make_unique<GamePresenter>(*m_gameSession, m_mainWindow.gameWidget());
}

void MainWindowPresenter::onConnectRequested(const QString& hostIp) {
	onShutdownRequested();

	// TODO: The server does not tell us our seat yet, so show the colours like a local game until it does.
	showLocalPlayers();

	auto session = std::make_unique<app::NetworkSession>();
	session->connect(hostIp.toStdString());

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_gameSession = std::move(session);
}

void MainWindowPresenter::onHostRequested(const GameConfig& config, const Player hostColour) {
	onShutdownRequested();
	showPlayers(hostColour, tr("Opponent"));

	auto session = std::make_unique<app::NetworkSession>();
	if (!session->host(config, hostColour)) {
		// TODO: Signal to the user that host failed.
		return;
	}

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
