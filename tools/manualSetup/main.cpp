#include "vision/gui/manualSetupPresenter.hpp"
#include "vision/gui/manualSetupView.hpp"

#include <QApplication>

//! Pick the board corners by hand, then run the rest of the vision pipeline on the live camera.
int main(int argc, char** argv) {
	QApplication application(argc, argv);

	tengen::vision::gui::ManualSetupView view;
	tengen::vision::gui::ManualSetupPresenter presenter(view, 0);
	view.resize(1400, 900);
	view.show();

	return application.exec();
}
