#pragma once

#include "gui/boardWidget.hpp"
#include "model/player.hpp"

#include <QLabel>
#include <QPushButton>
#include <QString>
#include <QTabWidget>
#include <QWidget>
#include <optional>

namespace tengen::gui {

class ChatWidget;
class PlayerCard;

class GameWidget : public QWidget {
	Q_OBJECT

public:
	explicit GameWidget(QWidget* parent = nullptr);
	~GameWidget() override;

	BoardWidget& boardWidget();
	ChatWidget& chatWidget();

	//! The player shows right of the status with playerColour, the opponent left with the other colour.
	//! Keeps the highlight as it is, so follow it with setCurrentPlayer().
	void setPlayers(const QString& playerName, const QString& opponentName, Player playerColour);
	void setCurrentPlayer(std::optional<Player> player); //!< Highlight the side to move. Nullopt if no player to move (game ended e.g.).
	void setGameStateText(const QString& text);
	void setChatEnabled(bool enabled);

signals:
	void passEvent();
	void resignEvent();

protected:
	void resizeEvent(QResizeEvent* event) override;

private:
	//! Initial setup constructing the layout of the window.
	void buildNetworkLayout();

private:
	QTabWidget* m_sideTabs   = nullptr; //!< Moves and chat, right of the board.
	ChatWidget* m_chatWidget = nullptr;
	int m_chatTabIndex       = -1;

	// Board area consists of Header, Board, Footer
	QWidget* m_boardArea = nullptr; //!< Header, board and footer. Kept as wide as the board is tall.

	QWidget* m_header          = nullptr; //!< Player cards and status above the board.
	PlayerCard* m_opponentCard = nullptr; //!< Left of the status.
	PlayerCard* m_ownCard      = nullptr; //!< Right of the status.
	QLabel* m_statusLabel      = nullptr; //!< Game status text (whose move, finished).

	BoardWidget* m_boardWidget = nullptr;

	QWidget* m_footer           = nullptr; //!< Pass and resign below the board.
	QPushButton* m_passButton   = nullptr;
	QPushButton* m_resignButton = nullptr;
};

} // namespace tengen::gui
