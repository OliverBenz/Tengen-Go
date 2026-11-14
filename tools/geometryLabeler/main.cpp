#include "labelPresenter.hpp"
#include "labelView.hpp"
#include "layoutPresenter.hpp"
#include "layoutView.hpp"

#include <QApplication>

//! Label photos with their geometry .json by clicking: open a folder, click the corners of every photo that has no
//! .json yet, submit. Then "Board layout..." writes the folder's board.txt from the stones clicked on one photo.
//! The file formats are described in tests/vision/resources/README.md.
int main(int argc, char** argv) {
	QApplication application(argc, argv);

	tengen::labeler::LabelView view;
	// Composition root: the label presenter knows nothing of the layout dialog.
	const tengen::labeler::LayoutRunner runLayout = [&view](const QImage& photo, const tengen::labeler::Geometry& geometry, const std::filesystem::path& file) {
		tengen::labeler::LayoutView dialog(&view);
		tengen::labeler::LayoutPresenter presenter(dialog, photo, geometry, file);
		return dialog.exec() == QDialog::Accepted;
	};
	tengen::labeler::LabelPresenter presenter(view, runLayout);
	view.resize(1400, 900);
	view.show();

	return application.exec();
}
