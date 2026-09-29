#pragma once

#include "core/gameRules.hpp"

#include <QWidget>

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;

namespace tengen::gui {

//! Little gui widget to select which game rules we want to play according to.
class RulesConfigWidget : public QWidget {
	Q_OBJECT

public:
	explicit RulesConfigWidget(QWidget* parent = nullptr); //!< Starts on Japanese rules.
	GameRules rules() const;                               //!< Get the rules the user picked.

private:
	QComboBox* m_ruleSet{nullptr}; //!< Selector for a standard ruleset, or custom rules as its last entry.

	QWidget* m_custom{nullptr};      //!< Holds the custom rule fields. Only shows while custom rules are picked.
	QComboBox* m_scoring{nullptr};   //!< Selector for how the game is scored.
	QComboBox* m_ko{nullptr};        //!< Selector for the ko rule.
	QDoubleSpinBox* m_komi{nullptr}; //!< Points white is given for playing second.
	QCheckBox* m_suicide{nullptr};   //!< Whether a player may capture their own stones.
};

} // namespace tengen::gui
