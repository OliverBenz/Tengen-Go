#pragma once

#include "engine/engineCatalog.hpp"
#include "model/gameRules.hpp"

#include <QDialog>

class QComboBox;
class QStackedWidget;

namespace tengen::gui {

class BoardSizeWidget;
class GnuGoConfigWidget;
class KataGoConfigWidget;
class PlayerColourWidget;
class RulesConfigWidget;

class BotDialog : public QDialog {
	Q_OBJECT

public:
	//! Offers every engine. The ones not installed show greyed out and cannot be picked.
	explicit BotDialog(const engine::InstalledEngines& engines, QWidget* parent = nullptr);

	unsigned boardSize() const;
	GameRules rules() const;                   //!< The rules the game is played under.
	engine::EngineConfig engineConfig() const; //!< The engine the user picked and how it should play.
	bool humanPlaysBlack() const;

private:
	//! Offer an engine along with the widget that configures it.
	void addEngine(const QString& name, QWidget* configWidget, bool installed);

private:
	QComboBox* m_engineCombo{nullptr}; //!< Selector for the engine type.

	QStackedWidget* m_engineConfigs{nullptr}; //!< Every engine's config widget, in the order m_engine offers the engines.
	GnuGoConfigWidget* m_gnuGo{nullptr};      //!< Configuration for the GnuGo engine.
	KataGoConfigWidget* m_kataGo{nullptr};    //!< Configuration for the KataGo engine.

	BoardSizeWidget* m_boardSize{nullptr}; //!< Selector for the board size.
	PlayerColourWidget* m_colour{nullptr}; //!< Selector for which colour stones the player uses.
	RulesConfigWidget* m_rules{nullptr};   //!< Selector for the rules.
};

} // namespace tengen::gui
