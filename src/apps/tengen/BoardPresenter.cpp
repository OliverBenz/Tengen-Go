#include "BoardPresenter.hpp"
#include "tengen/IGameSession.hpp"

#include <QMetaObject>
#include <QObject>

#include <cassert>

namespace tengen {

BoardPresenter::BoardPresenter(app::IGameSession& game, gui::BoardWidget& boardWidget) : m_game(game), m_boardWidget(boardWidget) {
	QObject::connect(&m_boardWidget, &gui::BoardWidget::boardEvent, this, &BoardPresenter::onBoardEvent);
	m_boardWidget.setBoard(m_game.board());
	m_boardWidget.setCurrentPlayer(m_game.currentPlayer());
	m_game.subscribe(this, app::AS_BoardChange | app::AS_PlayerChange);
}

BoardPresenter::~BoardPresenter() {
	m_game.unsubscribe(this);
}

void BoardPresenter::onAppEvent(const app::AppSignal signal) {
	auto* widget      = &m_boardWidget;
	switch (signal) {
	case app::AS_BoardChange: {
		const Board board = m_game.board();
		QMetaObject::invokeMethod(widget, [widget, board]() { widget->setBoard(board); }, Qt::QueuedConnection);
		return;
	}
	case app::AS_PlayerChange: {
		const auto player = m_game.currentPlayer();
		QMetaObject::invokeMethod(widget, [widget, player]() { widget->setCurrentPlayer(player); }, Qt::QueuedConnection);
		return;
	}
	default:
		return;
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
