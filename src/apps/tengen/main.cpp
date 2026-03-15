#include "MainWindow.hpp"
#include "MainWindowPresenter.hpp"

#include <QApplication>

int main(int argc, char* argv[]) {
	QApplication application(argc, argv);

	// Setup and show UI
	tengen::gui::MainWindow window;
	tengen::MainWindowPresenter presenter(window);
	window.resize(1200, 900);
	window.show();

	const auto exitCode = application.exec();
	return exitCode;
}
