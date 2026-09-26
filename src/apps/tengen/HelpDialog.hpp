#pragma once

#include <QDialog>

namespace tengen::gui {

//! The pages of the user documentation. They are HTML files in help/ next to the executable.
enum class HelpPage {
	Rules,
	Engine,
};

class HelpDialog : public QDialog {
	Q_OBJECT

public:
	explicit HelpDialog(HelpPage page, QWidget* parent = nullptr);
};

} // namespace tengen::gui
