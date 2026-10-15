#pragma once

#include "engine/gnuGoConfig.hpp"

#include <QWidget>

class QLabel;
class QSlider;

namespace tengen::gui {

//! Picks how GNU Go plays. Its strength is a level rather than a rank, so that is what the user picks.
class GnuGoConfigWidget : public QWidget {
	Q_OBJECT

public:
	//! Shows the config it is given. Only the level changes: the files are handed back as they came.
	explicit GnuGoConfigWidget(engine::GnuGoConfig config, QWidget* parent = nullptr);

	engine::GnuGoConfig config() const; //!< The config with the level the user picked.

private:
	engine::GnuGoConfig m_config;  //!< The config as it came in.
	QSlider* m_level{nullptr};     //!< Level at which the engine plays.
	QLabel* m_levelLabel{nullptr}; //!< Shows the level the slider currently sits on.
};

} // namespace tengen::gui
