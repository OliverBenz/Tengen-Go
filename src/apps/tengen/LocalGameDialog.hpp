#pragma once

#include "core/gameRules.hpp"

#include <QDialog>

class QComboBox;

namespace tengen::gui {

class RulesConfigWidget;

class LocalGameDialog : public QDialog {
	Q_OBJECT

public:
	explicit LocalGameDialog(QWidget* parent = nullptr);

	unsigned boardSize() const;
	GameRules rules() const; //!< The rules the game is played under.

private:
	QComboBox* m_boardSize{nullptr};     //!< Selector for the board size.
	RulesConfigWidget* m_rules{nullptr}; //!< Selector for the rules.
};

} // namespace tengen::gui
