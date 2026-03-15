#pragma once

#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QWidget>

#include <string>

namespace tengen::gui {

class ChatWidget : public QWidget {
	Q_OBJECT

public:
	explicit ChatWidget(QWidget* parent = nullptr);
	~ChatWidget() override;

	void appendMessage(const std::string& line);

signals:
	void chatEvent(const std::string& message);

private:
	void onSendClicked();

private:
	QListWidget* m_chatList = nullptr;
	QLineEdit* m_chatInput  = nullptr;
	QPushButton* m_chatSend = nullptr;
};

} // namespace tengen::gui
