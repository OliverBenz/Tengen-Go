#include "vision/gui/cameraPreviewView.hpp"

#include "imageView.hpp"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

namespace tengen::vision::gui {

static constexpr int IMAGE_SIZE = 960; //!< [px] Longer side of the shown image.

CameraPreviewView::CameraPreviewView(QWidget* parent, const Qt::WindowFlags flags)
    : QWidget(parent, flags) {
	m_devices     = new QComboBox(this);
	m_preview     = new QWidget(this);
	m_image       = new ImageView(m_preview);
	m_status      = new QLabel(m_preview);
	auto* refresh = new QToolButton(this);
	auto* redo    = new QPushButton(tr("Redo"), m_preview);
	m_devices->setPlaceholderText(tr("No devices found"));
	m_image->setFixedSize(IMAGE_SIZE, IMAGE_SIZE * 9 / 16); // Until the first frame tells.
	refresh->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
	refresh->setToolTip(tr("Refresh input devices."));

	auto* topRow = new QHBoxLayout();
	topRow->addStretch();
	topRow->addWidget(m_devices);
	topRow->addWidget(refresh);

	auto* bottomRow = new QHBoxLayout();
	bottomRow->addWidget(m_status, 1);
	bottomRow->addWidget(redo);

	auto* previewLayout = new QVBoxLayout(m_preview);
	previewLayout->setContentsMargins(0, 0, 0, 0);
	previewLayout->addWidget(m_image);
	previewLayout->addLayout(bottomRow);

	auto* layout = new QVBoxLayout(this);
	layout->setSizeConstraint(QLayout::SetFixedSize); // Follow the image size, also as a window.
	layout->addLayout(topRow);
	layout->addWidget(m_preview);

	connect(m_devices, &QComboBox::activated, this, &CameraPreviewView::deviceChosen); // User choices only.
	connect(refresh, &QToolButton::clicked, this, &CameraPreviewView::refreshRequested);
	connect(m_image, &ImageView::clicked, this, &CameraPreviewView::cornerPicked);
	connect(redo, &QPushButton::clicked, this, &CameraPreviewView::redoRequested);
}

void CameraPreviewView::showDevices(const QStringList& names, const int selected) {
	m_devices->clear();
	m_devices->addItems(names);
	m_devices->setCurrentIndex(selected);
	m_devices->setEnabled(!names.isEmpty());
	m_preview->setVisible(!names.isEmpty());
}

void CameraPreviewView::showFrame(const QImage& frame) {
	m_image->setFixedSize(frame.size().scaled(IMAGE_SIZE, IMAGE_SIZE, Qt::KeepAspectRatio));
	m_image->setImage(frame);
}

void CameraPreviewView::addCorner(const QPointF corner) {
	m_image->addMarker(corner);
}

void CameraPreviewView::clearCorners() {
	m_image->clearMarkers();
}

void CameraPreviewView::showStatus(const QString& text) {
	m_status->setText(text);
}

} // namespace tengen::vision::gui
