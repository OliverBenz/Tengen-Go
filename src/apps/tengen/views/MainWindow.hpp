#pragma once

#include "engine/engineCatalog.hpp"
#include "model/gameConfig.hpp"
#include "model/player.hpp"

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
	void botDialogRequested();                    //!< The presenter will look for engines before opening the dialog.
	void connectRequested(const QString& hostIp); //!< Join the game hosted at this address.
	void shutdownRequested();                     //!< The window closes. End the current game.

	void gameLocalRequested(const GameConfig& config);                                                               //!< Start a local game for two players on this machine.
	void gameBotRequested(const GameConfig& config, const engine::EngineConfig& engineConfig, bool humanPlaysBlack); //!< Start a game against this engine.
	void gameHostRequested(const GameConfig& config, Player hostColour);                                             //!< Host a network game and play it as hostColour.

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
