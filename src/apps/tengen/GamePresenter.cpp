#include "GamePresenter.hpp"

#include <QMetaObject>
#include <QObject>
#include <QString>

#include <cassert>

namespace tengen {

static QString currentPlayerText(const tengen::Player player) {
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

static QString gameStateText(const tengen::GameStatus status) {
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

GamePresenter::GamePresenter(app::SessionManager& game, gui::GameWidget& gameWidget) : m_game(game), m_gameWidget(gameWidget) {
	QObject::connect(&m_gameWidget, &gui::GameWidget::passEvent, &m_gameWidget, [this]() { this->onPassRequested(); });
	QObject::connect(&m_gameWidget, &gui::GameWidget::resignEvent, &m_gameWidget, [this]() { this->onResignRequested(); });

	m_boardPresenter = std::make_unique<BoardPresenter>(m_game, m_gameWidget.boardWidget());
	m_chatPresenter  = std::make_unique<ChatPresenter>(m_game, m_gameWidget.chatWidget());

	m_gameWidget.setCurrentPlayerText(currentPlayerText(m_game.currentPlayer()));
	m_gameWidget.setGameStateText(gameStateText(m_game.status()));

	m_game.subscribe(this, app::AS_PlayerChange | app::AS_StateChange);
}

GamePresenter::~GamePresenter() {
	m_game.unsubscribe(this);
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

void GamePresenter::onPassRequested() {
	m_game.tryPass();
}

void GamePresenter::onResignRequested() {
	m_game.tryResign();
}

} // namespace tengen
