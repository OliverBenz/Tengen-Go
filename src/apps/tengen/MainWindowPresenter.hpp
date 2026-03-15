#pragma once

#include "GamePresenter.hpp"
#include "MainWindow.hpp"
#include "tengen/sessionManager.hpp"

#include <memory>

namespace tengen {

class MainWindowPresenter {
public:
	explicit MainWindowPresenter(gui::MainWindow& mainWindow);
	~MainWindowPresenter();

private:
	gui::MainWindow& m_mainWindow;
	app::SessionManager m_game;
	std::unique_ptr<GamePresenter> m_gamePresenter = nullptr;
};

} // namespace tengen
