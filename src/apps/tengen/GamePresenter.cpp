#include "GamePresenter.hpp"

#include "BoardPresenter.hpp"
#include "ChatPresenter.hpp"
#include "gui/gameWidget.hpp"

#include <QMetaObject>
#include <QObject>
#include <QString>

#include <cassert>

namespace {

QString currentPlayerText(const tengen::Player player) {
	switch (player) {
	case tengen::Player::Black:
		return QStringLiteral("Current Player: Black");
	case tengen::Player::White:
		return QStringLiteral("Current Player: White");
	default:
		assert(false);
		return {};
	}
}

QString gameStateText(const tengen::GameStatus status) {
	switch (status) {
	case tengen::GameStatus::Idle:
		return QStringLiteral("Idle");
	case tengen::GameStatus::Ready:
		return QStringLiteral("Waiting for Player");
	case tengen::GameStatus::Active:
		return QStringLiteral("Active");
	case tengen::GameStatus::Done:
		return QStringLiteral("Game Finished");
	default:
		assert(false);
		return {};
	}
}

} // namespace

namespace tengen {

GamePresenter::GamePresenter(app::SessionManager& game, gui::GameWidget& gameWidget) : m_game(game), m_gameWidget(gameWidget) {
	// TODO: This should be done in boardPresenter?
	QObject::connect(&m_gameWidget.boardWidget(), &gui::BoardWidget::boardEvent, &m_gameWidget,
	                 [game = &m_game](const gui::BoardWidgetEvent& event) { dispatchBoardEvent(*game, event); });
	QObject::connect(&m_gameWidget, &gui::GameWidget::passEvent, &m_gameWidget, [game = &m_game]() { game->tryPass(); });
	QObject::connect(&m_gameWidget, &gui::GameWidget::resignEvent, &m_gameWidget, [game = &m_game]() { game->tryResign(); });

	m_boardPresenter = std::make_unique<BoardPresenter>(m_game, m_gameWidget.boardWidget());
	m_chatPresenter  = std::make_unique<ChatPresenter>(m_game, m_gameWidget.chatWidget());

	m_gameWidget.setCurrentPlayerText(currentPlayerText(m_game.currentPlayer()));
	m_gameWidget.setGameStateText(gameStateText(m_game.status()));

	m_game.subscribe(this, app::AS_PlayerChange | app::AS_StateChange);
	m_listenerRegistered = true;
}

GamePresenter::~GamePresenter() {
	if (m_listenerRegistered) {
		m_game.unsubscribe(this);
	}
}

void GamePresenter::onAppEvent(const app::AppSignal signal) {
	auto* widget = &m_gameWidget;
	switch (signal) {
	case app::AS_PlayerChange: {
		const auto text = currentPlayerText(m_game.currentPlayer());
		QMetaObject::invokeMethod(widget, [widget, text]() { widget->setCurrentPlayerText(text); }, Qt::QueuedConnection);
		return;
	}
	case app::AS_StateChange: {
		const auto text = gameStateText(m_game.status());
		QMetaObject::invokeMethod(widget, [widget, text]() { widget->setGameStateText(text); }, Qt::QueuedConnection);
		return;
	}
	default:
		return;
	}
}

void GamePresenter::dispatchBoardEvent(app::SessionManager& game, const gui::BoardWidgetEvent& event) {
	switch (event.type) {
	case gui::BoardWidgetEventType::Place:
		game.tryPlace(event.coord.x, event.coord.y);
		return;
	case gui::BoardWidgetEventType::Pass:
		game.tryPass();
		return;
	case gui::BoardWidgetEventType::Resign:
		game.tryResign();
		return;
	default:
		return;
	}
}

} // namespace tengen
