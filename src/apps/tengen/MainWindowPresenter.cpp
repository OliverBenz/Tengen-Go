#include "MainWindowPresenter.hpp"

#include <QObject>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow) : m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, &m_mainWindow, [this](const QString& hostIp) { m_game.connect(hostIp.toStdString()); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, &m_mainWindow, [this](const unsigned boardSize) { m_game.host(boardSize); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, &m_mainWindow, [this]() { m_game.shutdown(); });

	m_gamePresenter = std::make_unique<GamePresenter>(m_game, m_mainWindow.gameWidget());
}

MainWindowPresenter::~MainWindowPresenter() = default;

} // namespace tengen
