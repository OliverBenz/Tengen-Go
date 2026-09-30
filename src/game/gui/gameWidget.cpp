#include "gui/gameWidget.hpp"

#include "gui/chatWidget.hpp"
#include "playerCard.hpp"

#include <QHBoxLayout>
#include <QTabWidget>
#include <QVBoxLayout>

#include <algorithm>
#include <utility>

namespace tengen::gui {

GameWidget::GameWidget(QWidget* parent) : QWidget(parent) {
	setWindowTitle("Go Game");
	buildNetworkLayout();

	connect(m_passButton, &QPushButton::clicked, this, &GameWidget::passEvent);
	connect(m_resignButton, &QPushButton::clicked, this, &GameWidget::resignEvent);
}

GameWidget::~GameWidget() = default;

BoardWidget& GameWidget::boardWidget() {
	return *m_boardWidget;
}

ChatWidget& GameWidget::chatWidget() {
	return *m_chatWidget;
}

void GameWidget::setPlayers(const QString& playerName, const QString& opponentName, const Player playerColour) {
	m_ownCard->setPlayer(playerColour, playerName);
	m_opponentCard->setPlayer(opponent(playerColour), opponentName);
}

void GameWidget::setCurrentPlayer(const std::optional<Player> player) {
	m_opponentCard->setActive(player == m_opponentCard->colour());
	m_ownCard->setActive(player == m_ownCard->colour());
}

void GameWidget::setGameStateText(const QString& text) {
	m_statusLabel->setText(text);
}

void GameWidget::resizeEvent(QResizeEvent* event) {
	QWidget::resizeEvent(event); // The layout already placed everything for the new size.

	// The board is square. Any width beyond the height left for it would only open gaps above and below it,
	// pushing the header and buttons away. A maximum, unlike a fixed size, still lets the window shrink.
	const int boardHeight = m_boardArea->height() - m_header->height() - m_footer->height();
	m_boardArea->setMaximumWidth(std::max(boardHeight, m_boardArea->minimumSizeHint().width()));
}

void GameWidget::setChatEnabled(const bool enabled) {
	if (!m_sideTabs || !m_chatWidget || m_chatTabIndex < 0) {
		return;
	}

	if (!enabled && m_sideTabs->currentIndex() == m_chatTabIndex) {
		m_sideTabs->setCurrentIndex(0);
	}

	m_sideTabs->setTabEnabled(m_chatTabIndex, enabled);
	m_chatWidget->setEnabled(enabled);
}

void GameWidget::buildNetworkLayout() {
	static constexpr qreal STATUS_TEXT_SIZE_FACTOR = 1.2; //!< Scale up the status text to make more visible.

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->setContentsMargins(12, 12, 12, 12);
	mainLayout->setSpacing(8);

	auto* contentLayout = new QHBoxLayout();
	contentLayout->setSpacing(12);

	m_boardArea       = new QWidget(this);
	auto* boardColumn = new QVBoxLayout(m_boardArea);
	boardColumn->setContentsMargins(0, 0, 0, 0);
	boardColumn->setSpacing(0);

	m_header           = new QWidget(m_boardArea);
	auto* headerLayout = new QHBoxLayout(m_header);
	headerLayout->setContentsMargins(BoardWidget::BOARD_MARGIN, 4, BoardWidget::BOARD_MARGIN, 6); // Room for the cards' shadows.

	m_opponentCard = new PlayerCard(false, m_header);
	m_ownCard      = new PlayerCard(true, m_header);

	m_statusLabel = new QLabel("", m_header);
	m_statusLabel->setAlignment(Qt::AlignCenter);
	QFont statusFont = m_statusLabel->font();
	statusFont.setPointSizeF(statusFont.pointSizeF() * STATUS_TEXT_SIZE_FACTOR);
	m_statusLabel->setFont(statusFont);

	headerLayout->addWidget(m_opponentCard);
	headerLayout->addWidget(m_statusLabel, 1);
	headerLayout->addWidget(m_ownCard);
	boardColumn->addWidget(m_header);

	m_boardWidget = new BoardWidget(m_boardArea);
	m_boardWidget->setMinimumSize(640, 640);
	boardColumn->addWidget(m_boardWidget, 1);

	m_footer           = new QWidget(m_boardArea);
	auto* footerLayout = new QHBoxLayout(m_footer);
	footerLayout->setContentsMargins(BoardWidget::BOARD_MARGIN, 0, BoardWidget::BOARD_MARGIN, 0);

	m_passButton   = new QPushButton("Pass", m_footer);
	m_resignButton = new QPushButton("Resign", m_footer);
	footerLayout->addWidget(m_passButton);
	footerLayout->addWidget(m_resignButton);
	footerLayout->addStretch();
	boardColumn->addWidget(m_footer);

	// Takes the width until it is capped to the board's height (see resizeEvent), then the tabs get the rest.
	contentLayout->addWidget(m_boardArea, 1);

	m_sideTabs = new QTabWidget(this);

	auto* moveHistoryTab = new QWidget(m_sideTabs);
	auto* moveLayout     = new QVBoxLayout(moveHistoryTab);
	moveLayout->addWidget(new QLabel("Move history will be listed here.", moveHistoryTab));
	moveLayout->addStretch();
	m_sideTabs->addTab(moveHistoryTab, "Moves");

	auto* chatTab    = new QWidget(m_sideTabs);
	auto* chatLayout = new QVBoxLayout(chatTab);
	chatLayout->setContentsMargins(0, 0, 0, 0);
	m_chatWidget = new ChatWidget(chatTab);
	chatLayout->addWidget(m_chatWidget, 1);
	m_chatTabIndex = m_sideTabs->addTab(chatTab, "Chat");

	contentLayout->addWidget(m_sideTabs);
	mainLayout->addLayout(contentLayout, 1);

	setPlayers("White", "Black", Player::White);
}

} // namespace tengen::gui
