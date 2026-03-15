#include "ChatWidget.hpp"

#include <QAbstractItemView>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QVBoxLayout>

namespace tengen::gui {

ChatWidget::ChatWidget(QWidget* parent) : QWidget(parent) {
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);

	m_chatList = new QListWidget(this);
	m_chatList->setSelectionMode(QAbstractItemView::NoSelection);
	m_chatList->setFocusPolicy(Qt::NoFocus);
	layout->addWidget(m_chatList, 1);

	auto* chatInputRow    = new QWidget(this);
	auto* chatInputLayout = new QHBoxLayout(chatInputRow);
	chatInputLayout->setContentsMargins(0, 0, 0, 0);
	m_chatInput = new QLineEdit(chatInputRow);
	m_chatSend  = new QPushButton("Send", chatInputRow);
	chatInputLayout->addWidget(m_chatInput, 1);
	chatInputLayout->addWidget(m_chatSend);
	layout->addWidget(chatInputRow);

	connect(m_chatSend, &QPushButton::clicked, this, &ChatWidget::onSendClicked);
	connect(m_chatInput, &QLineEdit::returnPressed, this, &ChatWidget::onSendClicked);
}

ChatWidget::~ChatWidget() = default;

void ChatWidget::appendMessage(const std::string& line) {
	if (!m_chatList) {
		return;
	}
	m_chatList->addItem(QString::fromStdString(line));
	m_chatList->scrollToBottom();
}

void ChatWidget::onSendClicked() {
	if (!m_chatInput) {
		return;
	}

	const auto text = m_chatInput->text().trimmed();
	if (text.isEmpty()) {
		return;
	}

	emit chatEvent(text.toStdString());
	m_chatInput->clear();
}

} // namespace tengen::gui
