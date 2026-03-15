#pragma once

#include "gui/boardWidget.hpp"
#include "tengen/sessionManager.hpp"

namespace tengen {

class BoardPresenter : public app::IAppSignalListener {
public:
	BoardPresenter(app::SessionManager& game, gui::BoardWidget& boardWidget);
	~BoardPresenter() override;

	//! Called by the game thread. Ensure not blocking.
	void onAppEvent(app::AppSignal signal) override;

private:
	void dispatchBoardEvent(const gui::BoardWidgetEvent& event);

	app::SessionManager& m_game;
	gui::BoardWidget& m_boardWidget;
	bool m_listenerRegistered = false;
};

} // namespace tengen
