#include "BoardPresenter.hpp"

#include <QMetaObject>
#include <QObject>

#include <cassert>

namespace tengen {

BoardPresenter::BoardPresenter(app::SessionManager& game, gui::BoardWidget& boardWidget) : m_game(game), m_boardWidget(boardWidget) {
	QObject::connect(&m_boardWidget, &gui::BoardWidget::boardEvent, &m_boardWidget,
	                 [this](const gui::BoardWidgetEvent& event) { this->dispatchBoardEvent(event); });
	m_boardWidget.setBoard(m_game.board());
	m_game.subscribe(this, app::AS_BoardChange);
	m_listenerRegistered = true;
}

BoardPresenter::~BoardPresenter() {
	if (m_listenerRegistered) {
		m_game.unsubscribe(this);
	}
}

void BoardPresenter::onAppEvent(const app::AppSignal signal) {
	if (signal != app::AS_BoardChange) {
		return;
	}

	const Board board = m_game.board();
	auto* widget      = &m_boardWidget;
	QMetaObject::invokeMethod(widget, [widget, board]() { widget->setBoard(board); }, Qt::QueuedConnection);
}

void BoardPresenter::dispatchBoardEvent(const gui::BoardWidgetEvent& event) {
	switch (event.type) {
	case gui::BoardWidgetEventType::Place:
		m_game.tryPlace(event.coord.x, event.coord.y);
		break;
	case gui::BoardWidgetEventType::Pass:
		m_game.tryPass();
		break;
	case gui::BoardWidgetEventType::Resign:
		m_game.tryResign();
		break;
	default:
		assert(false);
		return;
	}
}

} // namespace tengen
