#include "HelpDialog.hpp"

#include <QCoreApplication>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileInfo>
#include <QTextBrowser>
#include <QUrl>
#include <QVBoxLayout>
#include <array>

namespace tengen::gui {

namespace {

struct PageInfo {
	const char* title; //!< Window title. Marked for translation, the dialog translates it.
	const char* file;  //!< File name inside help/.
};

// In the order of HelpPage.
static constexpr std::array pages = {
        PageInfo{QT_TR_NOOP("Rules of Go"), "rules.html"},
        PageInfo{QT_TR_NOOP("Bot Engines"), "engine.html"},
};

const PageInfo& infoOf(const HelpPage page) {
	return pages[static_cast<size_t>(page)];
}

QString documentPath(const PageInfo& info) {
	return QDir(QCoreApplication::applicationDirPath()).filePath(QString("help/") + info.file);
}

} // namespace

HelpDialog::HelpDialog(const HelpPage page, QWidget* parent)
    : QDialog(parent) {
	const PageInfo& info = infoOf(page);
	const QString title  = tr(info.title);
	setWindowTitle(title);
	resize(900, 600);

	auto* browser        = new QTextBrowser(this);
	const QString source = documentPath(info);
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
