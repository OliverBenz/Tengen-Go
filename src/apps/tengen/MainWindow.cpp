#include "MainWindow.hpp"

#include "ConnectDialog.hpp"
#include "HostDialog.hpp"
#include "gui/gameWidget.hpp"

#include <QMenuBar>
#include <cassert>

namespace tengen::gui {

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
	// Setup Window
	setWindowTitle("Go Game");
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
	auto* menu          = menuBar()->addMenu(tr("&Menu"));
	auto* connectAction = new QAction("&Connect to Server", this);
	auto* hostAction    = new QAction("&Host Server", this);
	menu->addAction(connectAction);
	menu->addAction(hostAction);
	connect(connectAction, &QAction::triggered, this, &MainWindow::openConnectDialog);
	connect(hostAction, &QAction::triggered, this, &MainWindow::openHostDialog);

	m_gameWidget = new GameWidget();
	setCentralWidget(m_gameWidget);
}

void MainWindow::openConnectDialog() {
	ConnectDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit connectRequested(dialog.ipAddress());
	}
}

void MainWindow::openHostDialog() {
	HostDialog dialog(this);

	if (dialog.exec() == QDialog::Accepted) {
		emit hostRequested(dialog.boardSize());
	}
}

void MainWindow::closeEvent(QCloseEvent* event) {
	emit shutdownRequested();
	QMainWindow::closeEvent(event);
}

} // namespace tengen::gui
