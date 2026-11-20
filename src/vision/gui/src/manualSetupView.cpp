#include "vision/gui/manualSetupView.hpp"

#include "imageView.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>

namespace tengen::vision::gui {

static constexpr int SIDE_COLUMN_WIDTH = 320; //!< [px]

ManualSetupView::ManualSetupView(QWidget* parent)
    : QWidget(parent) {
	setWindowTitle("Manual Board Setup");

	m_image     = new ImageView(this);
	m_rectified = new ImageView(this);
	m_status    = new QLabel(this);
	auto* apply = new QPushButton(tr("Apply"), this);
	auto* redo  = new QPushButton(tr("Redo"), this);
	m_rectified->setFixedWidth(SIDE_COLUMN_WIDTH);

	auto* controls = new QHBoxLayout();
	controls->addWidget(m_status, 1);
	controls->addWidget(apply);
	controls->addWidget(redo);

	auto* mainColumn = new QVBoxLayout();
	mainColumn->addWidget(m_image, 1);
	mainColumn->addLayout(controls);

	auto* layout = new QHBoxLayout(this);
	layout->addLayout(mainColumn, 1);
	layout->addWidget(m_rectified);

	connect(m_image, &ImageView::clicked, this, &ManualSetupView::cornerPicked);
	connect(apply, &QPushButton::clicked, this, &ManualSetupView::applyRequested);
	connect(redo, &QPushButton::clicked, this, &ManualSetupView::redoRequested);
}

void ManualSetupView::showImage(const QImage& image) {
	m_image->setImage(image);
}

void ManualSetupView::addCorner(const QPointF corner) {
	m_image->addMarker(corner);
}

void ManualSetupView::clearCorners() {
	m_image->clearMarkers();
}

void ManualSetupView::showRectified(const QImage& rectified) {
	m_rectified->setImage(rectified);
}

void ManualSetupView::showStatus(const QString& text) {
	m_status->setText(text);
}

} // namespace tengen::vision::gui
