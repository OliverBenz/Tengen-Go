#include "BoardPresenter.hpp"
#include "tengen/IGameSession.hpp"

#include <QMetaObject>
#include <QObject>

#include <cassert>

namespace tengen {

BoardPresenter::BoardPresenter(app::IGameSession& game, gui::BoardWidget& boardWidget) : m_game(game), m_boardWidget(boardWidget) {
	QObject::connect(&m_boardWidget, &gui::BoardWidget::boardEvent, this, &BoardPresenter::onBoardEvent);
	m_boardWidget.setBoard(m_game.board());
	m_game.subscribe(this, app::AS_BoardChange);
}

BoardPresenter::~BoardPresenter() {
	m_game.unsubscribe(this);
}

void BoardPresenter::onAppEvent(const app::AppSignalMask signal) {
	if (signal & app::AS_BoardChange) {
		m_boardWidget.setBoard(m_game.board());
	}
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
