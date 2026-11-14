#include "labelView.hpp"

#include "photoCanvas.hpp"

#include <QComboBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QPushButton>
#include <QShortcut>
#include <QVBoxLayout>

#include <utility>

namespace tengen::labeler {

LabelView::LabelView(QWidget* parent)
    : QWidget(parent) {
	setWindowTitle(tr("Geometry Labeler (mouse wheel: zoom, right drag: pan, arrow keys: move the last corner)"));

	m_canvas     = new PhotoCanvas(this);
	m_status     = new QLabel(this);
	m_boardSize  = new QComboBox(this);
	m_layout     = new QPushButton(tr("Board layout..."), this);
	auto* open   = new QPushButton(tr("Open folder..."), this);
	auto* undo   = new QPushButton(tr("Undo (Ctrl+Z)"), this);
	auto* skip   = new QPushButton(tr("Skip"), this);
	auto* submit = new QPushButton(tr("Submit (Enter)"), this);

	for (const unsigned size: {9u, 13u, 19u}) {
		m_boardSize->addItem(tr("%1x%1").arg(size), size);
	}
	undo->setShortcut(QKeySequence::Undo);
	submit->setShortcut(Qt::Key_Return);

	auto* controls = new QHBoxLayout();
	controls->addWidget(open);
	controls->addWidget(m_layout);
	controls->addWidget(m_status, 1);
	controls->addWidget(m_boardSize);
	controls->addWidget(undo);
	controls->addWidget(skip);
	controls->addWidget(submit);

	auto* layout = new QVBoxLayout(this);
	layout->addWidget(m_canvas, 1);
	layout->addLayout(controls);

	connect(open, &QPushButton::clicked, this, &LabelView::chooseFolder);
	connect(m_layout, &QPushButton::clicked, this, &LabelView::layoutRequested);
	connect(m_canvas, &PhotoCanvas::clicked, this, &LabelView::cornerPicked);
	connect(undo, &QPushButton::clicked, this, &LabelView::undoRequested);
	connect(skip, &QPushButton::clicked, this, &LabelView::skipRequested);
	connect(submit, &QPushButton::clicked, this, &LabelView::submitRequested);
	connect(m_boardSize, &QComboBox::currentIndexChanged, this, [this] { emit boardSizeChosen(m_boardSize->currentData().toUInt()); });

	const std::pair<Qt::Key, QPoint> arrows[] = {{Qt::Key_Left, {-1, 0}}, {Qt::Key_Right, {1, 0}}, {Qt::Key_Up, {0, -1}}, {Qt::Key_Down, {0, 1}}};
	for (const auto& [key, step]: arrows) {
		connect(new QShortcut(key, this), &QShortcut::activated, this, [this, step] { emit moveRequested(step); });
	}
}

void LabelView::showPhoto(const QImage& photo) {
	m_canvas->setPhoto(photo);
}

void LabelView::showCorners(const std::vector<QPoint>& boardCorners, const std::vector<QPoint>& gridCorners) {
	m_canvas->setCorners(boardCorners, gridCorners);
}

void LabelView::showStatus(const QString& text) {
	m_status->setText(text);
}

void LabelView::showBoardSize(const unsigned size) {
	m_boardSize->setCurrentIndex(m_boardSize->findData(size));
	m_canvas->setBoardSize(size);
}

void LabelView::showLayoutAvailable(const bool available, const QString& reason) {
	m_layout->setEnabled(available);
	m_layout->setToolTip(reason);
}

QString LabelView::choosePhoto(const QString& folder) {
	return QFileDialog::getOpenFileName(this, tr("Photo to click the stones on"), folder, tr("Photos (*.png *.jpg *.jpeg *.PNG *.JPG *.JPEG)"));
}

void LabelView::chooseFolder() {
	const QString folder = QFileDialog::getExistingDirectory(this, tr("Folder with photos"));
	if (!folder.isEmpty()) {
		emit folderChosen(folder);
	}
}

} // namespace tengen::labeler
