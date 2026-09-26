#include "MainWindow.hpp"

#include "BotDialog.hpp"
#include "ConnectDialog.hpp"
#include "HelpDialog.hpp"
#include "HostDialog.hpp"
#include "gui/gameWidget.hpp"

#include <QMenuBar>
#include <cassert>

namespace tengen::gui {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent) {
	// Setup Window
	setWindowTitle("Tengen Go");
	setWindowFlags(windowFlags() | Qt::Tool | Qt::WindowStaysOnTopHint);
	setAttribute(Qt::WA_QuitOnClose, true);
	buildLayout();
}

MainWindow::~MainWindow() = default;

GameWidget& MainWindow::gameWidget() {
	assert(m_gameWidget);
	return *m_gameWidget;
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
	connect(actNewLocalGame, &QAction::triggered, this, &MainWindow::gameLocalRequested); // Signal to signal connection
	connect(actNewBotGame, &QAction::triggered, this, &MainWindow::botDialogRequested);   // The presenter looks for the engines first.

	auto* network            = menuBar()->addMenu(tr("&Network"));
	auto* actConnectToServer = new QAction("&Connect to Server", this);
	auto* actHostServer      = new QAction("&Host Server", this);
	auto* actDisconnect      = new QAction("&Disconnect", this);
	network->addAction(actConnectToServer);
	network->addAction(actHostServer);
	network->addAction(actDisconnect);
	connect(actConnectToServer, &QAction::triggered, this, &MainWindow::openConnectDialog);
	connect(actHostServer, &QAction::triggered, this, &MainWindow::openHostDialog);

	auto* tools                   = menuBar()->addMenu(tr("&Tools"));
	auto* actImportBoardImage     = new QAction("&Import Board Image", this);
	auto* actStartCameraDetection = new QAction("&Start Camera Detection", this);
	auto* actCalibrateDetection   = new QAction("&Calibrate Detection", this);
	tools->addAction(actImportBoardImage);
	tools->addAction(actStartCameraDetection);
	tools->addAction(actCalibrateDetection);

	auto* help      = menuBar()->addMenu(tr("&Help"));
	auto* actRules  = new QAction("&Rules", this);
	auto* actEngine = new QAction("&Engine", this);
	auto* actAbout  = new QAction("&About", this);
	help->addAction(actRules);
	help->addAction(actEngine);
	help->addAction(actAbout);
	connect(actRules, &QAction::triggered, this, [this] { openHelp(HelpPage::Rules); });
	connect(actEngine, &QAction::triggered, this, [this] { openHelp(HelpPage::Engine); });

	m_gameWidget = new GameWidget();
	setCentralWidget(m_gameWidget);
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
		emit gameBotRequested(dialog.boardSize(), dialog.engineConfig(), dialog.humanPlaysBlack());
	}
}

void MainWindow::openHostDialog() {
	HostDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit hostRequested(dialog.boardSize());
	}
}

void MainWindow::openHelp(const HelpPage page) {
	HelpDialog dialog(page, this);
	dialog.exec();
}

void MainWindow::closeEvent(QCloseEvent* event) {
	emit shutdownRequested();
	QMainWindow::closeEvent(event);
}

} // namespace tengen::gui
