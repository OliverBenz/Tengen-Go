#pragma once

#include "gui/boardWidget.hpp"
#include <QLabel>
#include <QPushButton>
#include <QString>
#include <QTabWidget>
#include <QWidget>

namespace tengen::gui {

class ChatWidget;

class GameWidget : public QWidget {
	Q_OBJECT

public:
	explicit GameWidget(Board board, QWidget* parent = nullptr);
	~GameWidget() override;

	BoardWidget& boardWidget();
	ChatWidget& chatWidget();

	void setCurrentPlayerText(const QString& text);
	void setGameStateText(const QString& text);

signals:
	void passEvent();
	void resignEvent();

private:
	//! Initial setup constructing the layout of the window.
	void buildNetworkLayout(Board board);

private:
	BoardWidget* m_boardWidget = nullptr;
	ChatWidget* m_chatWidget   = nullptr;
	QTabWidget* m_sideTabs     = nullptr;

	QLabel* m_statusLabel     = nullptr; //!< Game status text (active, finished).
	QLabel* m_currPlayerLabel = nullptr; //!< Current player text.

	QPushButton* m_passButton   = nullptr;
	QPushButton* m_resignButton = nullptr;
};

} // namespace tengen::gui
