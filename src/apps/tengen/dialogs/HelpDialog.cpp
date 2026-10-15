#include "HelpDialog.hpp"

#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>
#include <map>
#include <string>

namespace tengen::gui {
namespace {

struct PageInfo {
	std::string title; //!< Window title.
	std::string file;  //!< File name inside help/.
};

const std::map<HelpPage, PageInfo> pages{
        {HelpPage::Rules, {"Rules of Go", "rules.html"}},
        {HelpPage::Engine, {"Bot Engines", "engine.html"}},
};

QString documentPath(const std::string& fileName) {
	return QDir(QCoreApplication::applicationDirPath()).filePath(QString::fromStdString("help/" + fileName));
}
} // namespace

HelpDialog::HelpDialog(const HelpPage page, QWidget* parent)
    : QDialog(parent) {
	// Get page information
	const PageInfo& info = pages.at(page);

	// Set page title
	const QString title = QString::fromStdString(info.title);
	setWindowTitle(title);
	resize(900, 600);

	// Setup document
	auto* browser        = new QTextBrowser(this);
	const QString source = documentPath(info.file);
	browser->setOpenExternalLinks(true); // We link to our sources.

	if (QFileInfo::exists(source)) {
		browser->setSource(QUrl::fromLocalFile(source));
	} else {
		browser->setHtml(tr("<h2>%1 unavailable</h2><p>Could not find <code>%2</code>.</p>").arg(title.toHtmlEscaped(), source.toHtmlEscaped()));
	}

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Close, this);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addWidget(browser);
	mainLayout->addWidget(buttons);
}

} // namespace tengen::gui
