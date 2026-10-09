#pragma once

#include "GamePresenter.hpp"
#include "MainWindow.hpp"
#include "engine/engineCatalog.hpp"
#include "model/gameConfig.hpp"
#include "tengen/IGameSession.hpp"

#include <QObject>
#include <memory>

namespace tengen {

class MainWindowPresenter : public QObject {
	Q_OBJECT

public:
	explicit MainWindowPresenter(gui::MainWindow& mainWindow);
	~MainWindowPresenter() override;

private slots:
	void onNewLocalGameRequested(const GameConfig& config);
	void onBotDialogRequested();
	void onNewBotGameRequested(const GameConfig& config, const engine::EngineConfig& engineConfig, bool humanPlaysBlack);
	void onConnectRequested(const QString& hostIp);
	void onHostRequested(const GameConfig& config, Player hostColour);
	void onShutdownRequested();

private:
	void startOpenPlay(const GameConfig& config);

	void showLocalPlayers();                                         //!< Local game: both sides are named by their colour.
	void showPlayers(Player ownColour, const QString& opponentName); //!< Set the player strings in the status box based given your stone colour and the opponent name. You get the name "You".

private:
	gui::MainWindow& m_mainWindow;                             //!< The main window.
	std::unique_ptr<app::IGameSession> m_gameSession{nullptr}; //!< The actual game.
	std::unique_ptr<GamePresenter> m_gamePresenter{nullptr};   //!< The 'drawer' of the game.
};

} // namespace tengen
