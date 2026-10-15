#pragma once

#include <QDialog>

namespace tengen::gui {

class AboutDialog : public QDialog {
	Q_OBJECT

public:
	explicit AboutDialog(QWidget* parent = nullptr);
};

} // namespace tengen::gui
