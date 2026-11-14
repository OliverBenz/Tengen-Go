#pragma once

#include "boardFile.hpp"

#include <QDialog>
#include <QImage>
#include <QString>

#include <vector>

class QLabel;

namespace tengen::labeler {

class BoardCanvas;

//! One photo seen from above to click its stones on. The B and W keys choose the colour a click places.
class LayoutView final : public QDialog {
	Q_OBJECT

public:
	explicit LayoutView(QWidget* parent = nullptr);

	void showBoard(const QImage& board, unsigned size);
	void showStones(const std::vector<Stone>& stones); //!< Row by row.
	void showStatus(const QString& text);

signals:
	void pointClicked(int row, int col);
	void colourChosen(Stone colour);
	void submitRequested();

private:
	BoardCanvas* m_canvas{nullptr};
	QLabel* m_status{nullptr};
};

} // namespace tengen::labeler
