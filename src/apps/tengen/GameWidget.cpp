#include "GameWidget.hpp"

#include "gui/chatWidget.hpp"

#include <QHBoxLayout>
#include <QTabWidget>
#include <QVBoxLayout>

#include <utility>

namespace tengen::gui {

GameWidget::GameWidget(Board board, QWidget* parent) : QWidget(parent) {
	setWindowTitle("Go Game");
	buildNetworkLayout(std::move(board));

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

void GameWidget::setCurrentPlayerText(const QString& text) {
	m_currPlayerLabel->setText(text);
}

void GameWidget::setGameStateText(const QString& text) {
	m_statusLabel->setText(text);
}

void GameWidget::buildNetworkLayout(Board board) {
	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->setContentsMargins(12, 12, 12, 12);
	mainLayout->setSpacing(8);

	m_statusLabel = new QLabel("", this);
	mainLayout->addWidget(m_statusLabel);

	auto* contentLayout = new QHBoxLayout();
	contentLayout->setSpacing(12);

	m_boardWidget = new BoardWidget(std::move(board), this);
	m_boardWidget->setMinimumSize(640, 640);
	contentLayout->addWidget(m_boardWidget);

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
	m_sideTabs->addTab(chatTab, "Chat");

	contentLayout->addWidget(m_sideTabs, 1);
	mainLayout->addLayout(contentLayout, 1);

	auto* footer       = new QWidget(this);
	auto* footerLayout = new QHBoxLayout(footer);
	footerLayout->setContentsMargins(0, 0, 0, 0);

	m_passButton   = new QPushButton("Pass", footer);
	m_resignButton = new QPushButton("Resign", footer);
	footerLayout->addWidget(m_passButton);
	footerLayout->addWidget(m_resignButton);

	m_currPlayerLabel = new QLabel("", footer);
	footerLayout->addWidget(m_currPlayerLabel);

	footer->setLayout(footerLayout);
	mainLayout->addWidget(footer);
}

} // namespace tengen::gui
