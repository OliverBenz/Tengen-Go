#pragma once

#include "model/player.hpp"

#include <QWidget>

class QButtonGroup;

namespace tengen::gui {

//! Little widget for selecting the colour the user plays. Ensures we always use the same input style.
class PlayerColourWidget : public QWidget {
	Q_OBJECT

public:
	explicit PlayerColourWidget(QWidget* parent = nullptr); //!< Starts on Black.
	Player player() const;                                  //!< Get the colour the user picked.

private:
	QButtonGroup* m_colours{nullptr}; //!< One button per colour. Each button's id is its Player value.
};

} // namespace tengen::gui
