#pragma once

#include "gui/boardWidget.hpp"
#include "gui/soundPlayer.hpp"
#include "tengen/IGameSession.hpp"

#include <QObject>

namespace tengen {

class BoardPresenter : public QObject, public app::IAppSignalListener {
	Q_OBJECT

public:
	BoardPresenter(app::IGameSession& game, gui::BoardWidget& boardWidget);
	~BoardPresenter() override;

	void onAppEvent(app::AppSignal signal) override; //!< Signalled by session thread. Offload work to GUI thread.

private slots:
	void onBoardEvent(const gui::BoardWidgetEvent& event);

private:
	void showBoard();           //!< Draw the session's board.
	void showCurrentPlayer();   //!< Show whose stone the next click places.
	void playStonePlaceSound(); //!< Play a sound when a stone is placed.

private:
	app::IGameSession& m_game;
	gui::BoardWidget& m_boardWidget;
	gui::SoundPlayer m_soundPlayer;
};

} // namespace tengen
