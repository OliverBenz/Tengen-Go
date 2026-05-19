#include "MainWindowPresenter.hpp"

#include "tengen/networkSession.hpp"

#include <QObject>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow) : m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, &m_mainWindow,
	                 [this](const QString& hostIp) { onConnectRequested(hostIp.toStdString()); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, &m_mainWindow, [this](const unsigned boardSize) { onHostRequested(boardSize); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, &m_mainWindow, [this]() { onShutdownRequested(); });
}

MainWindowPresenter::~MainWindowPresenter() = default;

void MainWindowPresenter::onConnectRequested(const std::string& hostIp) {
	if (m_game) {
		// Session already in progress
		return;
	}

	auto session = std::make_unique<app::NetworkSession>();
	session->connect(hostIp);

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_game = std::move(session);
}

void MainWindowPresenter::onHostRequested(const unsigned boardSize) {
	if (m_game) {
		// Session already in progress
		return;
	}

	auto session = std::make_unique<app::NetworkSession>();
	session->host(boardSize);

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_game = std::move(session);
}

void MainWindowPresenter::onShutdownRequested() {
	if (m_game) {
		m_gamePresenter = nullptr; // Destroy before m_game
		m_game->shutdown();
		m_game = nullptr;
	}
}

} // namespace tengen
