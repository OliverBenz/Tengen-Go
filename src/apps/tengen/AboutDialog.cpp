#include "AboutDialog.hpp"

#include <QApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QUrl>
#include <QVBoxLayout>

namespace tengen::gui {
namespace {

//! The installed license folder with NOTICES.md and the license texts.
//! Running from a build folder there is none, so link the repository's NOTICES.md instead.
QUrl licensesUrl() {
	const QString folder = QDir::cleanPath(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral(TENGEN_LICENSE_DIR)));
	if (QFileInfo::exists(folder + QStringLiteral("/NOTICES.md"))) {
		return QUrl::fromLocalFile(folder);
	}
	return QUrl(QStringLiteral("https://github.com/OliverBenz/Tengen-Go/blob/master/NOTICES.md"));
}

} // namespace

AboutDialog::AboutDialog(QWidget* parent)
    : QDialog(parent) {
	setWindowTitle(tr("About Tengen Go"));

	// The AGPL asks interactive programs to show the copyright, the license, the missing warranty, and where to read the license.
	auto* label = new QLabel(this);
	label->setTextFormat(Qt::RichText);
	label->setText(tr("<h2>Tengen Go</h2>"
	                  "<p>Version %1</p>"
	                  "<p>A modular Go platform bridging physical and digital games.</p>"
	                  "<p>Copyright (C) 2024 Oliver Benz</p>"
	                  "<p>Tengen is free software, licensed under the GNU Affero General Public License v3.0.<br>"
	                  "It comes with absolutely no warranty.</p>"
	                  "<p><a href=\"%2\">Licenses and third-party notices</a><br>"
	                  "<a href=\"https://github.com/OliverBenz/Tengen-Go\">Source code: github.com/OliverBenz/Tengen-Go</a></p>")
	                       .arg(QApplication::applicationVersion(), licensesUrl().toString()));
	label->setOpenExternalLinks(true);

	// About QT Button
	auto* aboutQt = new QPushButton(tr("About Qt"), this);
	connect(aboutQt, &QPushButton::clicked, this, [] { QApplication::aboutQt(); });

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* buttonRow = new QHBoxLayout;
	buttonRow->addWidget(aboutQt);
	buttonRow->addStretch();
	buttonRow->addWidget(buttons);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addWidget(label);
	mainLayout->addLayout(buttonRow);
}

} // namespace tengen::gui
