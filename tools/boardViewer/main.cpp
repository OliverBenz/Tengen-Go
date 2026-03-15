#include "core/serializer.hpp"
#include "gui/boardWidget.hpp"

#include <QApplication>
#include <QShortcut>

// Used for quickly visualising stuff I work on.
// Only supports visualising dotBW format for now. To be added: Real Images, openCV mats, custom serialization functions, sgf files.
int main(int argc, char* argv[]) {
	QApplication application(argc, argv);

	tengen::Board board(9u);
	if (!tengen::readBoard(std::filesystem::path{PATH_TEST_IMG} / "example.txt", board)) {
		return -1;
	}

	tengen::gui::BoardWidget boardWidget;
	boardWidget.setBoard(board);
	boardWidget.resize(800, 800);

	QShortcut escShortcut(QKeySequence(Qt::Key_Escape), &boardWidget);
	QObject::connect(&escShortcut, &QShortcut::activated, &application, &QApplication::quit);

	boardWidget.show();
	return application.exec();
}
