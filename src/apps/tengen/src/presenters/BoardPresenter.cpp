#include "BoardPresenter.hpp"
#include "tengen/IGameSession.hpp"

#include <QMetaObject>
#include <QObject>

#include <cassert>

namespace tengen {

BoardPresenter::BoardPresenter(app::IGameSession& game, gui::BoardWidget& boardWidget)
    : m_game(game), m_boardWidget(boardWidget) {
	QObject::connect(&m_boardWidget, &gui::BoardWidget::boardEvent, this, &BoardPresenter::onBoardEvent);

	// Subscribe before the first read: a change in between would otherwise go unseen.
	m_game.subscribe(this, app::AS_BoardChange | app::AS_PlayerChange | app::AS_ConnectionChange | app::AS_StonePlaced);
	showBoard();
	showGhostColour();
}

BoardPresenter::~BoardPresenter() {
	m_game.unsubscribe(this);
}

void BoardPresenter::onAppEvent(const app::AppSignalMask mask) {
	QMetaObject::invokeMethod(
	        this,
	        [this, mask]() {
		        if (mask & app::AS_BoardChange) {
			        showBoard();
		        }
		        if (mask & (app::AS_PlayerChange | app::AS_ConnectionChange)) {
			        showGhostColour();
		        }
		        if (mask & app::AS_StonePlaced) {
			        playStonePlaceSound();
		        }
	        },
	        Qt::QueuedConnection);
}

void BoardPresenter::showBoard() {
	m_boardWidget.setBoard(m_game.board());
}

void BoardPresenter::showGhostColour() {
	// Our own colour if we have one. Playing both sides (open play): whoever is to move.
	m_boardWidget.setGhostColour(m_game.localPlayer().value_or(m_game.currentPlayer()));
}

void BoardPresenter::playStonePlaceSound() {
	m_boardWidget.playStonePlaceSound();
}

void BoardPresenter::onBoardEvent(const gui::BoardWidgetEvent& event) {
	switch (event.type) {
	case gui::BoardWidgetEvent::Type::Place:
		m_game.tryPlace(event.coord.x, event.coord.y);
		break;
	case gui::BoardWidgetEvent::Type::Pass:
		m_game.tryPass();
		break;
	case gui::BoardWidgetEvent::Type::Resign:
		m_game.tryResign();
		break;
	default:
		assert(false);
		return;
	}
}

} // namespace tengen
