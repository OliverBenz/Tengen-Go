#pragma once

#include "model/player.hpp"

#include <QCloseEvent>
#include <QMainWindow>
#include <QString>

class QAction;

namespace tengen::gui {

class GameWidget;

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

	GameWidget& gameWidget();

	void setBotGameAvailable(bool available); //!< Bot games need an engine. Without one, the menu does not offer them.

signals:
	void gameLocalRequested();
	void gameBotRequested(unsigned boardSize, Skill botSkill, bool humanPlaysBlack);
	void connectRequested(const QString& hostIp);
	void hostRequested(unsigned boardSize);
	void shutdownRequested();

private:
	//! Initial setup constructing the layout of the window.
	void buildLayout();

private:
	void openBotDialog();
	void openConnectDialog();
	void openHostDialog();
	void openRulesDialog();

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	GameWidget* m_gameWidget = nullptr;
	QAction* m_botGameAction = nullptr; //!< Starts a bot game. Hidden while no engine is installed.
};

} // namespace tengen::gui
