#pragma once

#include "GamePresenter.hpp"
#include "MainWindow.hpp"
#include "tengen/IGameSession.hpp"

#include <memory>
#include <string>

namespace tengen {

class MainWindowPresenter {
public:
	explicit MainWindowPresenter(gui::MainWindow& mainWindow);
	~MainWindowPresenter();

private: // slots
	void onConnectRequested(const std::string& hostIp);
	void onHostRequested(const unsigned boardSize);
	void onShutdownRequested();

private:
	gui::MainWindow& m_mainWindow;
	std::unique_ptr<app::IGameSession> m_game{nullptr};
	std::unique_ptr<GamePresenter> m_gamePresenter{nullptr};
};

} // namespace tengen
