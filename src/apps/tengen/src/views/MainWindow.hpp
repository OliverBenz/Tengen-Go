#pragma once

#include "engine/engineCatalog.hpp"
#include "gui/resources.hpp"
#include "model/gameUpdates.hpp"
#include "model/player.hpp"

#include <QCloseEvent>
#include <QMainWindow>
#include <QString>

namespace tengen::gui {

class BoardWidget;
class GameWidget;
enum class HelpPage;

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(const Style& initialStyle, QWidget* parent = nullptr);
	~MainWindow() override;

	GameWidget& gameWidget();

	//! Let the user set a bot game up against one of the engines. Answers with gameBotRequested() once accepted.
	void openBotDialog(const engine::InstalledEngines& engines);

	//! Let the user pick a board style, starting from currentStyle. Answers with styleChangeRequested() once accepted.
	//! Emits styleDialogOpened() right before showing the dialog. Blocks until the dialog is closed.
	void openStyleDialog(const Style& currentStyle);

public slots:
	void applyStyle(const Style& style); //!< Apply the style to every widget that draws it.

signals:
	void botDialogRequested();                    //!< The presenter will look for engines before opening the dialog.
	void connectRequested(const QString& hostIp); //!< Join the game hosted at this address.
	void cameraPreviewRequested();                //!< Show the live camera image.
	void shutdownRequested();                     //!< The window closes. End the current game.

	void styleDialogRequested();                   //!< The presenter opens the dialog with the style in use.
	void styleDialogOpened(BoardWidget& preview);  //!< The dialog is about to show. Its preview board stays valid until openStyleDialog() returns.
	void styleChangeRequested(const Style& style); //!< Use this style from now on.

	void gameLocalRequested(const GameConfig& config);                                                               //!< Start a local game for two players on this machine.
	void gameBotRequested(const GameConfig& config, const engine::EngineConfig& engineConfig, bool humanPlaysBlack); //!< Start a game against this engine.
	void gameHostRequested(const GameConfig& config, Player hostColour);                                             //!< Host a network game and play it as hostColour.

private:
	//! Initial setup constructing the layout of the window.
	void buildLayout(const Style& initialStyle);

private:
	void openLocalGameDialog();
	void openConnectDialog();
	void openHostDialog();
	void openHelp(HelpPage page);
	void openAbout();

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	GameWidget* m_gameWidget = nullptr;
};

} // namespace tengen::gui
