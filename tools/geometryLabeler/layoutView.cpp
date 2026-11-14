#include "layoutView.hpp"

#include "boardCanvas.hpp"

#include <QDialogButtonBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QShortcut>
#include <QVBoxLayout>

namespace tengen::labeler {

LayoutView::LayoutView(QWidget* parent)
    : QDialog(parent) {
	setWindowTitle(tr("Board layout"));
	resize(900, 960);

	m_canvas      = new BoardCanvas(this);
	m_status      = new QLabel(this);
	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	buttons->button(QDialogButtonBox::Ok)->setText(tr("Write board.txt (Enter)"));

	auto* controls = new QHBoxLayout();
	controls->addWidget(m_status, 1);
	controls->addWidget(buttons);

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(m_canvas, 1);
	layout->addLayout(controls);

	connect(m_canvas, &BoardCanvas::clicked, this, &LayoutView::pointClicked);
	connect(buttons, &QDialogButtonBox::accepted, this, &LayoutView::submitRequested);
	connect(buttons, &QDialogButtonBox::rejected, this, &LayoutView::reject);
	connect(new QShortcut(Qt::Key_B, this), &QShortcut::activated, this, [this] { emit colourChosen(Stone::black); });
	connect(new QShortcut(Qt::Key_W, this), &QShortcut::activated, this, [this] { emit colourChosen(Stone::white); });
}

void LayoutView::showBoard(const QImage& board, const unsigned size) {
	m_canvas->setBoard(board, size);
}

void LayoutView::showStones(const std::vector<Stone>& stones) {
	m_canvas->setStones(stones);
}

void LayoutView::showStatus(const QString& text) {
	m_status->setText(text);
}

} // namespace tengen::labeler
