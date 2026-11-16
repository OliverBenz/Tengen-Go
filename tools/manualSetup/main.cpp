#include "vision/gui/manualSetupPresenter.hpp"
#include "vision/gui/manualSetupView.hpp"

#include <QApplication>
#include <QCoreApplication>

//! Pick the board corners by hand, then watch the grid found in them and the stones on the live camera.
int main(int argc, char** argv) {
	QApplication application(argc, argv);

	tengen::vision::gui::ManualSetupView view;
	tengen::vision::gui::ManualSetupPresenter presenter(view, 0, QCoreApplication::applicationDirPath().toStdString() + "/models/stoneNet.onnx");
	view.resize(1400, 900);
	view.show();

	return application.exec();
}
