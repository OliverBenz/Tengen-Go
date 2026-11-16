#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "engine/engineCatalog.hpp"
#include "model/gameUpdates.hpp"
#include "vision/gui/manualSetupView.hpp"

#include <QCoreApplication>
#include <QObject>
#include <QStandardPaths>
#include <cassert>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tengen {

static constexpr int CAMERA_INDEX = 0; //!< The system's default camera.

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow)
    : QObject(nullptr), m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameBotRequested, this, &MainWindowPresenter::onNewBotGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameHostRequested, this, &MainWindowPresenter::onHostRequested);

	QObject::connect(&m_mainWindow, &gui::MainWindow::botDialogRequested, this, &MainWindowPresenter::onBotDialogRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, this, &MainWindowPresenter::onConnectRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, this, &MainWindowPresenter::onShutdownRequested);

	QObject::connect(&m_mainWindow, &gui::MainWindow::visionPreviewRequested, this, &MainWindowPresenter::onVisionPreviewRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::visionPreviewClosed, this, &MainWindowPresenter::onVisionPreviewClosed);

	startOpenPlay(GameConfig{.boardSize = 9u, .rules = fromRuleSet(RuleSet::Japanese)});
}

MainWindowPresenter::~MainWindowPresenter() = default;

void MainWindowPresenter::startOpenPlay(const GameConfig& config) {
	resetPresenters();
	showLocalPlayers();
	m_sessions.startOpen(config);
	createPresenters();
}

void MainWindowPresenter::createPresenters() {
	assert(m_sessions.game()); // Only right after a session started.

	m_gamePresenter = std::make_unique<GamePresenter>(*m_sessions.game(), m_mainWindow.gameWidget());
	if (auto* chat = m_sessions.chat()) {
		m_gamePresenter->addChatWindow(*chat);
	}
	if (auto* network = m_sessions.network()) {
		m_networkPresenter = std::make_unique<NetworkPresenter>(*network, m_mainWindow.gameWidget());
	}
}

void MainWindowPresenter::resetPresenters() {
	m_gamePresenter.reset();
	m_networkPresenter.reset();
}

void MainWindowPresenter::showLocalPlayers() {
	m_mainWindow.gameWidget().setPlayers(tr("White"), tr("Black"), Player::White);
}

void MainWindowPresenter::showPlayers(const Player ownColour, const QString& opponentName) {
	m_mainWindow.gameWidget().setPlayers(tr("You"), opponentName, ownColour);
}

void MainWindowPresenter::onNewLocalGameRequested(const GameConfig& config) {
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
	resetPresenters();
	showPlayers(humanPlaysBlack ? Player::Black : Player::White, tr("Bot"));
	m_sessions.startBot(config, engineConfig, humanPlaysBlack);
	createPresenters();
}

void MainWindowPresenter::onConnectRequested(const QString& hostIp) {
	resetPresenters();
	showLocalPlayers(); // Until the welcome tells us our colour.
	m_sessions.startNetwork();
	createPresenters(); // Before connecting, so they miss nothing the host sends.
	m_sessions.connect(hostIp.toStdString());
}

void MainWindowPresenter::onHostRequested(const GameConfig& config, const Player hostColour) {
	resetPresenters();
	showPlayers(hostColour, tr("Opponent"));
	m_sessions.startNetwork();
	createPresenters(); // Before hosting, so they miss nothing the host sends.
	if (!m_sessions.host(config, hostColour)) {
		// TODO: Signal to the user that host failed.
		onShutdownRequested();
	}
}

void MainWindowPresenter::onShutdownRequested() {
	resetPresenters();
	m_sessions.shutdown();
}

void MainWindowPresenter::onVisionPreviewRequested() {
	if (!m_visionPreview) { // Else it is open already.
		// The build puts the models next to our executable.
		const std::filesystem::path stoneModel = (QCoreApplication::applicationDirPath() + "/models/stoneNet.onnx").toStdWString();
		try {
			m_visionPreview = std::make_unique<vision::gui::ManualSetupPresenter>(m_mainWindow.visionPreview(), CAMERA_INDEX, stoneModel);
		} catch (const std::runtime_error&) {
			m_mainWindow.visionPreview().showStatus(tr("The stone model could not be loaded."));
		}
	}
	m_mainWindow.showVisionPreview();
}

void MainWindowPresenter::onVisionPreviewClosed() {
	m_visionPreview.reset(); // Stops the camera.
}

} // namespace tengen
