#include "MainWindow.hpp"
#include "MainWindowPresenter.hpp"
#include "gui/styleSettings.hpp"
#include "gui/styleStore.hpp"

#include <QApplication>
#include <QSettings>
#include <QStandardPaths>

int main(int argc, char* argv[]) {
	QApplication application(argc, argv);
	QApplication::setApplicationName("tengen"); // Names our data folders, whatever the executable is called.
	QApplication::setApplicationVersion(TENGEN_VERSION);

	// Same readable file on every platform, instead of e.g. the Windows registry.
	QSettings settings(QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation) + "/settings.ini", QSettings::IniFormat);
	tengen::gui::StyleStore styleStore(tengen::gui::loadStyle(settings));
	QObject::connect(&styleStore, &tengen::gui::StyleStore::styleChanged, &styleStore, [&settings](const tengen::gui::Style& style) {
		tengen::gui::saveStyle(settings, style);
	});

	// Setup and show UI
	tengen::gui::MainWindow window(styleStore.style());
	tengen::MainWindowPresenter presenter(window, styleStore);
	window.resize(1200, 900);
	window.show();

	const auto exitCode = application.exec();
	return exitCode;
}
