#pragma once

#include <QWidget>

class QButtonGroup;

namespace tengen::gui {

//! Little widget for selecting the board size we want to play on. Ensures we always use the same input style.
class BoardSizeWidget : public QWidget {
	Q_OBJECT

public:
	explicit BoardSizeWidget(QWidget* parent = nullptr); //!< Starts on 9x9.
	unsigned boardSize() const;                          //!< Get the board size the user picked.

private:
	QButtonGroup* m_sizes{nullptr}; //!< One button per size. Each button's id is its size.
};

} // namespace tengen::gui
