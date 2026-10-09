#include "GamePresenter.hpp"
#include "tengen/IGameSession.hpp"

#include <QMetaObject>
#include <QObject>
#include <QString>

#include <cassert>

namespace tengen {

static QString gameStateText(const tengen::GameStatus status, const tengen::Player player) {
	switch (status) {
	case tengen::GameStatus::Idle:
		return QStringLiteral("Idle");
	case tengen::GameStatus::Ready:
		return QStringLiteral("Waiting for Player");
	case tengen::GameStatus::Active:
		return player == tengen::Player::Black ? QStringLiteral("Black to move") : QStringLiteral("White to move");
	case tengen::GameStatus::Done:
		return QStringLiteral("Game Finished");
	default:
		assert(false);
		return {};
	}
}

GamePresenter::GamePresenter(app::IGameSession& game, gui::GameWidget& gameWidget) : m_game(game), m_gameWidget(gameWidget) {
	QObject::connect(&m_gameWidget, &gui::GameWidget::passEvent, this, &GamePresenter::onPassRequested);
	QObject::connect(&m_gameWidget, &gui::GameWidget::resignEvent, this, &GamePresenter::onResignRequested);

	m_boardPresenter = std::make_unique<BoardPresenter>(m_game, m_gameWidget.boardWidget());
	m_gameWidget.setChatEnabled(false);

	m_game.subscribe(this, app::AS_PlayerChange | app::AS_StateChange); // Subscribe before the first read
	showStatus();
}

GamePresenter::~GamePresenter() {
	m_chatPresenter.reset();
	m_boardPresenter.reset();
	m_game.unsubscribe(this);
}

void GamePresenter::addChatWindow(app::IChatSession& chat) {
	m_chatPresenter = std::make_unique<ChatPresenter>(chat, m_gameWidget.chatWidget());
	m_gameWidget.setChatEnabled(true);
}

void GamePresenter::onAppEvent(const app::AppSignal signal) {
	switch (signal) {
	case app::AS_PlayerChange:
	case app::AS_StateChange:
		QMetaObject::invokeMethod(this, [this]() { showStatus(); }, Qt::QueuedConnection);
		return;
	default:
		return;
	}
}

void GamePresenter::showStatus() {
	const auto status = m_game.status();
	const auto player = m_game.currentPlayer();

	m_gameWidget.setGameStateText(gameStateText(status, player));
	m_gameWidget.setCurrentPlayer(status == GameStatus::Active ? std::optional{player} : std::nullopt);
}

void GamePresenter::onPassRequested() {
	m_game.tryPass();
}

void GamePresenter::onResignRequested() {
	m_game.tryResign();
}

} // namespace tengen
