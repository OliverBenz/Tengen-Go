#pragma once

#include "model/gameConfig.hpp"
#include "model/player.hpp"

#include <QDialog>

namespace tengen::gui {

class BoardSizeWidget;
class PlayerColourWidget;
class RulesConfigWidget;

class HostDialog : public QDialog {
	Q_OBJECT

public:
	explicit HostDialog(QWidget* parent = nullptr);

	GameConfig config() const; //!< How the game is to be played.
	Player hostColour() const; //!< The colour the host plays.

private:
	BoardSizeWidget* m_boardSize{nullptr}; //!< Selector for the board size.
	PlayerColourWidget* m_colour{nullptr}; //!< Selector for which colour stones the host uses.
	RulesConfigWidget* m_rules{nullptr};   //!< Selector for the rules.
};

} // namespace tengen::gui
