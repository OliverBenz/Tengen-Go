#pragma once

#include <QDialog>

namespace tengen::gui {

class BoardSizeWidget;

class HostDialog : public QDialog {
	Q_OBJECT

public:
	explicit HostDialog(QWidget* parent = nullptr);

	unsigned boardSize() const;

private:
	BoardSizeWidget* m_boardSize{nullptr}; //!< Selector for the board size.
};

} // namespace tengen::gui
