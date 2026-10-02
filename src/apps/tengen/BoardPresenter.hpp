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

	void onAppEvent(app::AppSignal signal) override; //!< Called by the game thread. Ensure not blocking.

private slots:
	void onBoardEvent(const gui::BoardWidgetEvent& event);

private:
	app::IGameSession& m_game;
	gui::BoardWidget& m_boardWidget;
	gui::SoundPlayer m_soundPlayer;
};

} // namespace tengen
