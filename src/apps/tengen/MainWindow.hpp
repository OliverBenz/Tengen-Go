#pragma once

#include "GamePresenter.hpp"
#include "tengen/sessionManager.hpp"

#include <QMainWindow>
#include <memory>

namespace tengen::gui {

class GameWidget;

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

private:
	//! Initial setup constructing the layout of the window.
	void buildLayout();

private: // Slots
	void openConnectDialog();
	void openHostDialog();
	void closeEvent(QCloseEvent* event);

private:
	app::SessionManager m_game;

	QWidget* m_menuWidget;
	GameWidget* m_gameWidget                                 = nullptr;
	std::unique_ptr<::tengen::GamePresenter> m_gamePresenter = nullptr;
};

} // namespace tengen::gui
