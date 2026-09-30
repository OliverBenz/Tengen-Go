#pragma once

#include "model/gameRules.hpp"
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

	unsigned boardSize() const;
	Player hostColour() const; //!< The colour the host plays.
	GameRules rules() const;   //!< The rules the game is played under.

private:
	BoardSizeWidget* m_boardSize{nullptr}; //!< Selector for the board size.
	PlayerColourWidget* m_colour{nullptr}; //!< Selector for which colour stones the host uses.
	RulesConfigWidget* m_rules{nullptr};   //!< Selector for the rules.
};

} // namespace tengen::gui
