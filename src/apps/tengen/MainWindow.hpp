#pragma once

#include "engine/engineCatalog.hpp"
#include "model/gameRules.hpp"

#include <QCloseEvent>
#include <QMainWindow>
#include <QString>

namespace tengen::gui {

class GameWidget;
enum class HelpPage;

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

	GameWidget& gameWidget();

	//! Let the user set a bot game up against one of the engines. Answers with gameBotRequested() once accepted.
	void openBotDialog(const engine::InstalledEngines& engines);

signals:
	void gameLocalRequested(unsigned boardSize, const GameRules& rules);
	void botDialogRequested(); //!< The user wants a bot game. Answer with openBotDialog().
	void gameBotRequested(unsigned boardSize, const GameRules& rules, const engine::EngineConfig& engineConfig, bool humanPlaysBlack);
	void connectRequested(const QString& hostIp);
	void hostRequested(unsigned boardSize, const GameRules& rules);
	void shutdownRequested();

private:
	//! Initial setup constructing the layout of the window.
	void buildLayout();

private:
	void openLocalGameDialog();
	void openConnectDialog();
	void openHostDialog();
	void openSettingsStyle();
	void openHelp(HelpPage page);
	void openAbout();

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	GameWidget* m_gameWidget = nullptr;
};

} // namespace tengen::gui
