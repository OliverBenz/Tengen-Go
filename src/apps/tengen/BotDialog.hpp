#pragma once

#include "engine/engineCatalog.hpp"

#include <QDialog>

class QComboBox;
class QStackedWidget;

namespace tengen::gui {

class GnuGoConfigWidget;
class KataGoConfigWidget;

class BotDialog : public QDialog {
	Q_OBJECT

public:
	//! Offers every engine. The ones not installed show greyed out and cannot be picked.
	explicit BotDialog(const engine::InstalledEngines& engines, QWidget* parent = nullptr);

	unsigned boardSize() const;
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

	QComboBox* m_boardSize{nullptr}; //!< Slector for the board size.
	QComboBox* m_colour{nullptr};    //!< Selector for which colour stones the player uses.
};

} // namespace tengen::gui
