#pragma once

#include "BoardPresenter.hpp"
#include "ChatPresenter.hpp"
#include "gui/boardWidget.hpp"
#include "tengen/sessionManager.hpp"
#include <QCloseEvent>
#include <QLabel>
#include <QPushButton>
#include <QTabWidget>
#include <QWidget>
#include <memory>

namespace tengen::gui {

class ChatWidget;

class GameWidget : public QWidget, public app::IAppSignalListener {
	Q_OBJECT

public:
	explicit GameWidget(app::SessionManager& game, QWidget* parent = nullptr);
	~GameWidget() override;

	//! Called by the game thread. Ensure not blocking.
	void onAppEvent(app::AppSignal signal) override;

private:
	//! Initial setup constructing the layout of the window.
	void buildNetworkLayout();

	void setCurrentPlayerText(); //!< Get current player from game and update the label.
	void setGameStateText();     //!< Get game state from game and update the label.

private: // Slots
	void onBoardWidgetEvent(const BoardWidgetEvent& event);
	void onPassClicked();
	void onResignClicked();

private:
	app::SessionManager& m_game;

	BoardWidget* m_boardWidget                       = nullptr;
	std::unique_ptr<BoardPresenter> m_boardPresenter = nullptr;
	ChatWidget* m_chatWidget                         = nullptr;
	std::unique_ptr<ChatPresenter> m_chatPresenter   = nullptr;
	QTabWidget* m_sideTabs                           = nullptr;

	QLabel* m_statusLabel     = nullptr; //!< Game status text (active, finished).
	QLabel* m_currPlayerLabel = nullptr; //!< Current player text.

	QPushButton* m_passButton   = nullptr;
	QPushButton* m_resignButton = nullptr;
};

} // namespace tengen::gui
