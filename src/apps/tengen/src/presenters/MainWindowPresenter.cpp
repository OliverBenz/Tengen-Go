#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "engine/engineCatalog.hpp"
#include "model/gameUpdates.hpp"
#include "vision/gui/cameraPreviewPresenter.hpp"
#include "vision/gui/cameraPreviewView.hpp"
#include "vision/gui/cameraSource.hpp"

#include <QCoreApplication>
#include <QObject>
#include <QStandardPaths>
#include <cassert>
#include <filesystem>
#include <memory>
#include <vector>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow, gui::StyleStore& styleStore)
    : QObject(nullptr), m_mainWindow(mainWindow), m_styleStore(styleStore) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameLocalRequested, this, &MainWindowPresenter::onNewLocalGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameBotRequested, this, &MainWindowPresenter::onNewBotGameRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::gameHostRequested, this, &MainWindowPresenter::onHostRequested);

	QObject::connect(&m_mainWindow, &gui::MainWindow::botDialogRequested, this, &MainWindowPresenter::onBotDialogRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, this, &MainWindowPresenter::onConnectRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::cameraPreviewRequested, this, &MainWindowPresenter::onCameraPreviewRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, this, &MainWindowPresenter::onShutdownRequested);

	QObject::connect(&m_mainWindow, &gui::MainWindow::styleDialogRequested, this, &MainWindowPresenter::onStyleDialogRequested);
	QObject::connect(&m_mainWindow, &gui::MainWindow::styleChangeRequested, &m_styleStore, &gui::StyleStore::setStyle);
	QObject::connect(&m_styleStore, &gui::StyleStore::styleChanged, &m_mainWindow, &gui::MainWindow::applyStyle);

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

void MainWindowPresenter::onCameraPreviewRequested() {
	if (!m_cameraPreview) { // Else it is open already.
		m_cameraPreview = new vision::gui::CameraPreviewView(&m_mainWindow, Qt::Window);
		m_cameraPreview->setAttribute(Qt::WA_DeleteOnClose); // With the camera and presenter in it.
		m_cameraPreview->setWindowTitle(tr("Camera Preview"));
		auto* camera    = new vision::gui::CameraSource(m_cameraPreview);
		auto* presenter = new vision::gui::CameraPreviewPresenter(*m_cameraPreview, *camera);
		presenter->setParent(m_cameraPreview);
	}
	m_cameraPreview->show();
	m_cameraPreview->raise();
	m_cameraPreview->activateWindow();
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

void MainWindowPresenter::onStyleDialogRequested() {
	// The preview is a game of its own, so the running one stays untouched. It lives as long as the dialog is open.
	app::SessionManager previewSession;
	std::unique_ptr<BoardPresenter> previewPresenter;

	const auto opened = QObject::connect(&m_mainWindow, &gui::MainWindow::styleDialogOpened, this, [&](gui::BoardWidget& preview) {
		previewSession.startOpen(GameConfig{.boardSize = 19u, .rules = fromRuleSet(RuleSet::Japanese)});
		previewPresenter = std::make_unique<BoardPresenter>(*previewSession.game(), preview);
	});
	m_mainWindow.openStyleDialog(m_styleStore.style()); // Blocks until the dialog closes.
	QObject::disconnect(opened);
}

void MainWindowPresenter::onShutdownRequested() {
	resetPresenters();
	m_sessions.shutdown();
}

} // namespace tengen
