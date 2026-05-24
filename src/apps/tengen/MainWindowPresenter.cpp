#include "MainWindowPresenter.hpp"

#include "GamePresenter.hpp"
#include "tengen/networkSession.hpp"
#include "tengen/openSession.hpp"

#include <QObject>
#include <memory>

namespace tengen {

MainWindowPresenter::MainWindowPresenter(gui::MainWindow& mainWindow) : m_mainWindow(mainWindow) {
	QObject::connect(&m_mainWindow, &gui::MainWindow::connectRequested, &m_mainWindow,
	                 [this](const QString& hostIp) { onConnectRequested(hostIp.toStdString()); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::hostRequested, &m_mainWindow, [this](const unsigned boardSize) { onHostRequested(boardSize); });
	QObject::connect(&m_mainWindow, &gui::MainWindow::shutdownRequested, &m_mainWindow, [this]() { onShutdownRequested(); });

	startOpenPlay();
}

MainWindowPresenter::~MainWindowPresenter() {
	onShutdownRequested();
}

void MainWindowPresenter::startOpenPlay() {
	m_game          = std::make_unique<app::OpenSession>(9u, m_dispatcher);
	m_gamePresenter = std::make_unique<GamePresenter>(*m_game, m_mainWindow.gameWidget());
}

void MainWindowPresenter::onConnectRequested(const std::string& hostIp) {
	onShutdownRequested();

	auto session = std::make_unique<app::NetworkSession>(m_dispatcher);
	session->connect(hostIp);

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_game = std::move(session);
}

void MainWindowPresenter::onHostRequested(const unsigned boardSize) {
	onShutdownRequested();

	auto session = std::make_unique<app::NetworkSession>(m_dispatcher);
	session->host(boardSize);

	auto& game      = static_cast<app::IGameSession&>(*session);
	auto& chat      = static_cast<app::IChatSession&>(*session);
	m_gamePresenter = std::make_unique<GamePresenter>(game, m_mainWindow.gameWidget());
	m_gamePresenter->addChatWindow(chat);
	m_game = std::move(session);
}

void MainWindowPresenter::onShutdownRequested() {
	if (m_game) {
		m_gamePresenter.reset(); // Destroy before m_game
		m_game->shutdown();
		m_dispatcher.flush();
		m_game.reset();
	}
}

} // namespace tengen
