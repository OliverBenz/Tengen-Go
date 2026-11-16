#include "MainWindow.hpp"

#include "AboutDialog.hpp"
#include "BoardStyleDialog.hpp"
#include "BotDialog.hpp"
#include "ConnectDialog.hpp"
#include "HelpDialog.hpp"
#include "HostDialog.hpp"
#include "LocalGameDialog.hpp"
#include "gui/gameWidget.hpp"
#include "gui/resources.hpp"
#include "vision/gui/manualSetupView.hpp"

#include <QDialog>
#include <QMenuBar>
#include <QVBoxLayout>
#include <cassert>

namespace tengen::gui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
	// Setup Window
	setWindowTitle("Tengen Go");
	setAttribute(Qt::WA_QuitOnClose, true);
	buildLayout();
}

MainWindow::~MainWindow() = default;

GameWidget& MainWindow::gameWidget() {
	assert(m_gameWidget);
	return *m_gameWidget;
}

vision::gui::ManualSetupView& MainWindow::visionPreview() {
	assert(m_visionPreview);
	return *m_visionPreview;
}

void MainWindow::buildLayout() {
	// Menu Bar
	auto* game            = menuBar()->addMenu(tr("&Game"));
	auto* actNewLocalGame = new QAction("&New Local Game", this);
	auto* actNewBotGame   = new QAction("New &Bot Game", this);
	auto* actSaveGame     = new QAction("&Save Game", this);
	auto* actLoadGame     = new QAction("&Load Game", this);
	game->addAction(actNewLocalGame);
	game->addAction(actNewBotGame);
	game->addAction(actSaveGame);
	game->addAction(actLoadGame);
	connect(actNewLocalGame, &QAction::triggered, this, &MainWindow::openLocalGameDialog);
	connect(actNewBotGame, &QAction::triggered, this, &MainWindow::botDialogRequested); // The presenter looks for the engines first.

	auto* network            = menuBar()->addMenu(tr("&Network"));
	auto* actConnectToServer = new QAction("&Connect to Server", this);
	auto* actHostServer      = new QAction("&Host Server", this);
	auto* actDisconnect      = new QAction("&Disconnect", this);
	network->addAction(actConnectToServer);
	network->addAction(actHostServer);
	network->addAction(actDisconnect);
	connect(actConnectToServer, &QAction::triggered, this, &MainWindow::openConnectDialog);
	connect(actHostServer, &QAction::triggered, this, &MainWindow::openHostDialog);

	auto* vision     = menuBar()->addMenu(tr("&Vision"));
	auto* actPreview = new QAction("&Preview", this);
	vision->addAction(actPreview);
	connect(actPreview, &QAction::triggered, this, &MainWindow::visionPreviewRequested);
	// auto* actImportBoardImage     = new QAction("&Import Board Image", this);
	// auto* actStartCameraDetection = new QAction("&Start Camera Detection", this);
	// auto* actCalibrateDetection   = new QAction("&Calibrate Detection", this);
	// vision->addAction(actImportBoardImage);
	// vision->addAction(actStartCameraDetection);
	// vision->addAction(actCalibrateDetection);

	auto* settings = menuBar()->addMenu(tr("&Settings"));
	auto* actStyle = new QAction("&Style", this);
	settings->addAction(actStyle);
	connect(actStyle, &QAction::triggered, this, &MainWindow::openSettingsStyle);

	auto* help      = menuBar()->addMenu(tr("&Help"));
	auto* actRules  = new QAction("&Rules", this);
	auto* actEngine = new QAction("&Engine", this);
	auto* actAbout  = new QAction("&About", this);
	help->addAction(actRules);
	help->addAction(actEngine);
	help->addAction(actAbout);
	connect(actRules, &QAction::triggered, this, [this] { openHelp(HelpPage::Rules); });
	connect(actEngine, &QAction::triggered, this, [this] { openHelp(HelpPage::Engine); });
	connect(actAbout, &QAction::triggered, this, &MainWindow::openAbout);

	m_gameWidget = new GameWidget();
	setCentralWidget(m_gameWidget);
	m_gameWidget->boardWidget().setBackgroundTexture(boardStyle::defaultTexture());

	buildVisionPreview();
}

void MainWindow::buildVisionPreview() {
	m_visionWindow  = new QDialog(this);
	m_visionPreview = new vision::gui::ManualSetupView(m_visionWindow);
	m_visionWindow->setWindowTitle(tr("Vision Preview"));
	m_visionWindow->resize(1100, 700);

	auto* layout = new QVBoxLayout(m_visionWindow);
	layout->setContentsMargins(0, 0, 0, 0); // The view brings its own.
	layout->addWidget(m_visionPreview);

	connect(m_visionWindow, &QDialog::finished, this, &MainWindow::visionPreviewClosed);
}

void MainWindow::openLocalGameDialog() {
	LocalGameDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit gameLocalRequested(dialog.config());
	}
}

void MainWindow::openConnectDialog() {
	ConnectDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit connectRequested(dialog.ipAddress());
	}
}

void MainWindow::openBotDialog(const engine::InstalledEngines& engines) {
	BotDialog dialog(engines, this);

	if (dialog.exec() == QDialog::Accepted) {
		emit gameBotRequested(dialog.config(), dialog.engineConfig(), dialog.humanPlaysBlack());
	}
}

void MainWindow::showVisionPreview() {
	m_visionWindow->show();
	m_visionWindow->raise();
	m_visionWindow->activateWindow();
}

void MainWindow::openHostDialog() {
	HostDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit gameHostRequested(dialog.config(), dialog.hostColour());
	}
}

void MainWindow::openSettingsStyle() {
	auto& board = m_gameWidget->boardWidget();
	BoardStyleDialog dialog(board.backgroundTexture(), board.showCoordinates(), this);

	if (dialog.exec() == QDialog::Accepted) {
		board.setBackgroundTexture(dialog.texture());
		board.setShowCoordinates(dialog.showCoordinates());
	}
}

void MainWindow::openHelp(const HelpPage page) {
	HelpDialog dialog(page, this);
	dialog.exec();
}

void MainWindow::openAbout() {
	AboutDialog dialog(this);
	dialog.exec();
}

void MainWindow::closeEvent(QCloseEvent* event) {
	emit shutdownRequested();
	QMainWindow::closeEvent(event);
}

} // namespace tengen::gui
