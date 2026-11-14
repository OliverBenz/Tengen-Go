#pragma once

#include <QImage>
#include <QPoint>
#include <QString>
#include <QWidget>

#include <vector>

class QComboBox;
class QLabel;
class QPushButton;

namespace tengen::labeler {

class PhotoCanvas;

//! One photo to click the corners on, the board size, and the buttons to move through the folder.
class LabelView final : public QWidget {
	Q_OBJECT

public:
	explicit LabelView(QWidget* parent = nullptr);

	void showPhoto(const QImage& photo);                                                               //!< A null image clears it.
	void showCorners(const std::vector<QPoint>& boardCorners, const std::vector<QPoint>& gridCorners); //!< In photo pixels.
	void showStatus(const QString& text);
	void showBoardSize(unsigned size);                               //!< One of the sizes offered.
	void showLayoutAvailable(bool available, const QString& reason); //!< The reason is the board layout button's tooltip.
	QString choosePhoto(const QString& folder);                      //!< Empty if the user cancels.

signals:
	void folderChosen(const QString& folder);
	void cornerPicked(QPoint pixel);
	void undoRequested();
	void moveRequested(QPoint step); //!< Move the last corner by `step` photo pixels.
	void skipRequested();
	void boardSizeChosen(unsigned size);
	void submitRequested();
	void layoutRequested();

private:
	void chooseFolder();

private:
	PhotoCanvas* m_canvas{nullptr};
	QLabel* m_status{nullptr};
	QComboBox* m_boardSize{nullptr};
	QPushButton* m_layout{nullptr};
};

} // namespace tengen::labeler
