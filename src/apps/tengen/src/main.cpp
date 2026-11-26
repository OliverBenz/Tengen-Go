#include "MainWindow.hpp"
#include "MainWindowPresenter.hpp"
#include "gui/styleStore.hpp"

#include <QApplication>

int main(int argc, char* argv[]) {
	QApplication application(argc, argv);
	QApplication::setApplicationName("tengen"); // Names our data folders, whatever the executable is called.
	QApplication::setApplicationVersion(TENGEN_VERSION);

	// Setup and show UI
	tengen::gui::StyleStore styleStore;
	tengen::gui::MainWindow window(styleStore.style());
	tengen::MainWindowPresenter presenter(window, styleStore);
	window.resize(1200, 900);
	window.show();

	const auto exitCode = application.exec();
	return exitCode;
}
