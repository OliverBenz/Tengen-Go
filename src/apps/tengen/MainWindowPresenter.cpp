#include "MainWindowPresenter.hpp"

#include "tengen/networkSession.hpp"

#include <QObject>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow) : m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, &m_mainWindow, [this](const QString& hostIp) { onConnectRequested(hostIp.toStdString()); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, &m_mainWindow, [this](const unsigned boardSize) { onHostRequested(boardSize); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, &m_mainWindow, [this]() { onShutdownRequested(); });

	m_gamePresenter = std::make_unique<GamePresenter>(m_game, m_mainWindow.gameWidget());
}

MainWindowPresenter::~MainWindowPresenter() = default;

void MainWindowPresenter::onConnectRequested(const std::string& hostIp) {
    if(m_game) {
        // Session already in progress
        return;
    }

    m_game = std::make_unique<app::NetworkSession>();
    m_game->connect(hostIp);
}

void MainWindowPresenter::onHostRequested(const unsigned boardSize) {
    if(m_game) {
        // Session already in progress
        return;
    }

    m_game = std::make_unique<app::NetworkSession>();
    m_game->host(boardSize);
}

void MainWindowPresenter::onShutdownRequested() {
    if(m_game){
        m_game->shutdown();
        m_game = nullptr;
    }
}

} // namespace tengen
