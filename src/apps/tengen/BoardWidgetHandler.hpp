#pragma once

#include "gui/boardWidget.hpp"
#include "tengen/sessionManager.hpp"

namespace tengen::gui {

class BoardWidgetHandler : public app::IAppSignalListener {
public:
	BoardWidgetHandler(app::SessionManager& game, BoardWidget& boardWidget);
	~BoardWidgetHandler() override;

	//! Called by the game thread. Ensure not blocking.
	void onAppEvent(app::AppSignal signal) override;

private:
	app::SessionManager& m_game;
	BoardWidget& m_boardWidget;
	bool m_listenerRegistered = false;
};

} // namespace tengen::gui
