#include "vision/gui/cameraPreviewView.hpp"

#include <QComboBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>
#include <QStyle>
#include <QToolButton>
#include <QVBoxLayout>

namespace tengen::vision::gui {

static constexpr int IMAGE_SIZE = 960; //!< [px] Longer side of the shown image.

CameraPreviewView::CameraPreviewView(QWidget* parent)
    : QWidget(parent) {
	m_devices     = new QComboBox(this);
	m_image       = new QLabel(this);
	auto* refresh = new QToolButton(this);
	m_devices->setPlaceholderText(tr("No devices found"));
	m_image->setFixedSize(IMAGE_SIZE, IMAGE_SIZE * 9 / 16); // Until the first frame tells.
	refresh->setIcon(style()->standardIcon(QStyle::SP_BrowserReload));
	refresh->setToolTip(tr("Refresh input devices."));

	auto* topRow = new QHBoxLayout();
	topRow->addStretch();
	topRow->addWidget(m_devices);
	topRow->addWidget(refresh);

	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addLayout(topRow);
	layout->addWidget(m_image);

	connect(m_devices, &QComboBox::activated, this, &CameraPreviewView::deviceChosen); // User choices only.
	connect(refresh, &QToolButton::clicked, this, &CameraPreviewView::refreshRequested);
}

void CameraPreviewView::showDevices(const QStringList& names, const int selected) {
	m_devices->clear();
	m_devices->addItems(names);
	m_devices->setCurrentIndex(selected);
	m_devices->setEnabled(!names.isEmpty());
	m_image->setVisible(!names.isEmpty());
}

void CameraPreviewView::showFrame(const QImage& frame) {
	const QPixmap pixmap = QPixmap::fromImage(frame.scaled(IMAGE_SIZE, IMAGE_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation));
	m_image->setFixedSize(pixmap.size());
	m_image->setPixmap(pixmap);
}

} // namespace tengen::vision::gui
