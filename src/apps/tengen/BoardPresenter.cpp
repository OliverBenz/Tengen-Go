#include "BoardPresenter.hpp"

#include <QMetaObject>

namespace tengen {

BoardPresenter::BoardPresenter(app::SessionManager& game, gui::BoardWidget& boardWidget) : m_game(game), m_boardWidget(boardWidget) {
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

} // namespace tengen
