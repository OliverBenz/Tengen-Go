#include "BoardPresenter.hpp"
#include "tengen/IGameSession.hpp"

#include <QMetaObject>
#include <QObject>

#include <cassert>

namespace tengen {

BoardPresenter::BoardPresenter(app::IGameSession& game, gui::BoardWidget& boardWidget) : m_game(game), m_boardWidget(boardWidget) {
	QObject::connect(&m_boardWidget, &gui::BoardWidget::boardEvent, this, &BoardPresenter::onBoardEvent);

	// Subscribe before the first read: a change in between would otherwise go unseen.
	m_game.subscribe(this, app::AS_BoardChange | app::AS_PlayerChange | app::AS_StonePlaced);
	showBoard();
	showCurrentPlayer();
}

BoardPresenter::~BoardPresenter() {
	m_game.unsubscribe(this);
}

void BoardPresenter::onAppEvent(const app::AppSignal signal) {
	switch (signal) {
	case app::AS_BoardChange:
		QMetaObject::invokeMethod(this, [this]() { showBoard(); }, Qt::QueuedConnection);
		return;
	case app::AS_PlayerChange:
		QMetaObject::invokeMethod(this, [this]() { showCurrentPlayer(); }, Qt::QueuedConnection);
		return;
	case app::AS_StonePlaced:
		QMetaObject::invokeMethod(this, [this]() { playStonePlaceSound(); }, Qt::QueuedConnection);
		return;
	default:
		return;
	}
}

void BoardPresenter::showBoard() {
	m_boardWidget.setBoard(m_game.board());
}

void BoardPresenter::showCurrentPlayer() {
	m_boardWidget.setCurrentPlayer(m_game.currentPlayer());
}

void BoardPresenter::playStonePlaceSound() {
	m_soundPlayer.playStonePlace();
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
