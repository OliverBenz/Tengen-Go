#include "AboutDialog.hpp"

#include <QApplication>
#include <QDialogButtonBox>
#include <QLabel>
#include <QVBoxLayout>

namespace tengen::gui {

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent) {
	setWindowTitle(tr("About Tengen Go"));

	auto* label = new QLabel(this);
	label->setTextFormat(Qt::RichText);
	label->setText(tr("<h2>Tengen Go</h2>"
	                  "<p>Version %1</p>"
	                  "<p>A modular Go platform bridging physical and digital games.</p>"
	                  "<p><a href=\"https://github.com/OliverBenz/Tengen-Go\">github.com/OliverBenz/Tengen-Go</a></p>")
	                       .arg(QApplication::applicationVersion()));
	label->setOpenExternalLinks(true);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addWidget(label);
	mainLayout->addWidget(buttons);
}

} // namespace tengen::gui
