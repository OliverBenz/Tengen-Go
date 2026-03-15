#pragma once

#include "gui/boardWidget.hpp"
#include "tengen/sessionManager.hpp"

namespace tengen {

class BoardPresenter : public app::IAppSignalListener {
public:
	BoardPresenter(app::SessionManager& game, gui::BoardWidget& boardWidget);
	~BoardPresenter() override;

	void onAppEvent(app::AppSignal signal) override; //!< Called by the game thread. Ensure not blocking.

private:
	void onBoardEvent(const gui::BoardWidgetEvent& event);

	app::SessionManager& m_game;
	gui::BoardWidget& m_boardWidget;
};

} // namespace tengen
