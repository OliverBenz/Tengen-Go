#include "GameWidget.hpp"

#include "ChatWidget.hpp"

#include <QHBoxLayout>
#include <QMetaObject>
#include <QTabWidget>
#include <QVBoxLayout>
#include <format>
#include <iostream>

namespace tengen::gui {

GameWidget::GameWidget(app::SessionManager& game, QWidget* parent) : QWidget(parent), m_game{game} {
	// Setup Window
	setWindowTitle("Go Game");
	buildNetworkLayout();

	// Connect slots
	connect(m_boardWidget, &BoardWidget::boardEvent, this, &GameWidget::onBoardWidgetEvent);
	connect(m_passButton, &QPushButton::clicked, this, &GameWidget::onPassClicked);
	connect(m_resignButton, &QPushButton::clicked, this, &GameWidget::onResignClicked);
	m_boardPresenter = std::make_unique<BoardPresenter>(m_game, *m_boardWidget);
	m_chatPresenter  = std::make_unique<ChatPresenter>(m_game, *m_chatWidget);

	// Setup Game Stuff
	setCurrentPlayerText();
	setGameStateText();
	m_game.subscribe(this, app::AS_PlayerChange | app::AS_StateChange);
}

GameWidget::~GameWidget() {
	m_game.unsubscribe(this);
}

void GameWidget::onAppEvent(const app::AppSignal signal) {
	switch (signal) {
	case app::AS_PlayerChange:
		QMetaObject::invokeMethod(this, [this]() { setCurrentPlayerText(); }, Qt::QueuedConnection);
		break;
	case app::AS_StateChange:
		QMetaObject::invokeMethod(this, [this]() { setGameStateText(); }, Qt::QueuedConnection);
		break;
	default:
		break;
	}
}


void GameWidget::buildNetworkLayout() {
	auto* central = new QWidget(this);

	// Layout Window top to bottom
	auto* mainLayout = new QVBoxLayout(central);
	mainLayout->setContentsMargins(12, 12, 12, 12);
	mainLayout->setSpacing(8);

	// Top: Label
	m_statusLabel = new QLabel("", central);
	mainLayout->addWidget(m_statusLabel);

	// Middle: Content section
	auto* contentLayout = new QHBoxLayout();
	contentLayout->setSpacing(12);

	m_boardWidget = new BoardWidget(m_game.board(), central);
	m_boardWidget->setMinimumSize(640, 640);
	contentLayout->addWidget(m_boardWidget);

	// Tabs on right side
	m_sideTabs = new QTabWidget(central);

	// Move History
	auto* moveHistoryTab = new QWidget(m_sideTabs);
	auto* moveLayout     = new QVBoxLayout(moveHistoryTab); // Title and content below
	moveLayout->addWidget(new QLabel("Move history will be listed here.", moveHistoryTab));
	moveLayout->addStretch(); // Fill space below Label
	m_sideTabs->addTab(moveHistoryTab, "Moves");

	// Chat
	auto* chatTab    = new QWidget(m_sideTabs);
	auto* chatLayout = new QVBoxLayout(chatTab); // Title and content below
	chatLayout->setContentsMargins(0, 0, 0, 0);
	m_chatWidget = new ChatWidget(chatTab);
	chatLayout->addWidget(m_chatWidget, 1);
	m_sideTabs->addTab(chatTab, "Chat");

	contentLayout->addWidget(m_sideTabs, 1);
	mainLayout->addLayout(contentLayout, 1);

	// Bottom: Footer
	auto* footer       = new QWidget(central);
	auto* footerLayout = new QHBoxLayout(footer);
	footerLayout->setContentsMargins(0, 0, 0, 0);

	// Buttons
	m_passButton   = new QPushButton("Pass", footer);
	m_resignButton = new QPushButton("Resign", footer);
	footerLayout->addWidget(m_passButton);
	footerLayout->addWidget(m_resignButton);

	// Label
	m_currPlayerLabel = new QLabel("", footer);
	footerLayout->addWidget(m_currPlayerLabel);

	footer->setLayout(footerLayout);
	mainLayout->addWidget(footer);

	central->setLayout(mainLayout);
}

void GameWidget::setCurrentPlayerText() {
	const auto text = std::format("Current Player: {}", m_game.currentPlayer() == Player::Black ? "Black" : "White");
	m_currPlayerLabel->setText(QString::fromStdString(text));
}

void GameWidget::setGameStateText() {
	static const std::map<GameStatus, std::string> message{{GameStatus::Idle, "Idle"},
	                                                       {GameStatus::Ready, "Waiting for Player"},
	                                                       {GameStatus::Active, "Active"},
	                                                       {GameStatus::Done, "Game Finished"}};
	assert(message.contains(m_game.status()));
	m_statusLabel->setText(QString::fromStdString(message.at(m_game.status())));
}

void GameWidget::onPassClicked() {
	m_game.tryPass();
}

void GameWidget::onBoardWidgetEvent(const BoardWidgetEvent& event) {
	switch (event.type) {
	case BoardWidgetEventType::Place:
		m_game.tryPlace(event.coord.x, event.coord.y);
		return;
	case BoardWidgetEventType::Pass:
		m_game.tryPass();
		return;
	case BoardWidgetEventType::Resign:
		m_game.tryResign();
		return;
	default:
		return;
	}
}

void GameWidget::onResignClicked() {
	m_game.tryResign();
}

} // namespace tengen::gui
