#pragma once

#include "engine/kataGoConfig.hpp"

#include <QWidget>

class QLabel;
class QSlider;

namespace tengen::gui {

//! Picks how KataGo plays: the rank of the human it imitates.
class KataGoConfigWidget : public QWidget {
	Q_OBJECT

public:
	//! Shows the config it is given. Only the rank changes: the files are handed back as they came.
	explicit KataGoConfigWidget(engine::KataGoConfig config, QWidget* parent = nullptr);

	engine::KataGoConfig config() const; //!< The config with the rank the user picked.

private:
	engine::KataGoConfig m_config; //!< The config as it came in.
	QSlider* m_rank{nullptr};      //!< The difficulty rank at which the engine plays.
	QLabel* m_rankLabel{nullptr};  //!< Shows the rank the slider currently sits on.
};

} // namespace tengen::gui
