#pragma once

#include "model/gameConfig.hpp"

#include <QDialog>

namespace tengen::gui {

class BoardSizeWidget;
class RulesConfigWidget;

class LocalGameDialog : public QDialog {
	Q_OBJECT

public:
	explicit LocalGameDialog(QWidget* parent = nullptr);

	GameConfig config() const; //!< How the game is to be played.

private:
	BoardSizeWidget* m_boardSize{nullptr}; //!< Selector for the board size.
	RulesConfigWidget* m_rules{nullptr};   //!< Selector for the rules.
};

} // namespace tengen::gui
