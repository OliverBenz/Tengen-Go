#include "vision/gui/manualSetupPresenter.hpp"

#include "vision/gui/manualSetupView.hpp"

#include <QMetaObject>

#include <utility>

namespace tengen::vision::gui {

//! A copy that owns its pixels, so it outlives the image.
static QImage toQImage(const Image& image) {
	return QImage(image.rgb.data(), image.width, image.height, qsizetype{3} * image.width, QImage::Format_RGB888).copy();
}

ManualSetupPresenter::ManualSetupPresenter(ManualSetupView& view, const int cameraIndex)
    : m_view(view), m_setup(cameraIndex, callbacks()) {
	connect(&m_view, &ManualSetupView::cornerPicked, this, &ManualSetupPresenter::onCornerPicked);
	connect(&m_view, &ManualSetupView::applyRequested, this, &ManualSetupPresenter::onApplyRequested);
	connect(&m_view, &ManualSetupView::redoRequested, this, &ManualSetupPresenter::onRedoRequested);
	showPicking();
}

// The images are converted on the camera thread, so the GUI thread only has to draw them.
ManualSetup::Callbacks ManualSetupPresenter::callbacks() {
	const auto onGuiThread = [this](auto job) { QMetaObject::invokeMethod(this, std::move(job), Qt::QueuedConnection); };
	return {
	        .onFrame      = [onGuiThread, this](const Image& frame) { onGuiThread([this, image = toQImage(frame)] { showFrame(image); }); },
	        .onAnalysed   = [onGuiThread, this](const BoardSnapshot& snapshot) { onGuiThread([this, warped = toQImage(snapshot.warped), rectified = toQImage(snapshot.rectified), size = snapshot.boardSize] { showSnapshot(warped, rectified, size); }); },
	        .onCameraLost = [onGuiThread, this] { onGuiThread([this] { m_view.showStatus(tr("No camera, or it stopped delivering.")); }); },
	};
}

void ManualSetupPresenter::onCornerPicked(const QPointF corner) {
	if (m_setup.pick(static_cast<float>(corner.x()), static_cast<float>(corner.y()))) {
		m_view.addCorner(corner);
	}
}

void ManualSetupPresenter::onApplyRequested() {
	if (!m_setup.apply()) {
		m_view.showStatus(tr("Click all 4 corners first."));
		return;
	}
	m_view.clearCorners();
	m_view.showStatus(tr("Looking for the grid..."));
}

void ManualSetupPresenter::onRedoRequested() {
	m_setup.redo();
	showPicking();
}

void ManualSetupPresenter::showPicking() {
	m_view.showImage({}); // The warped image is no frame to pick corners on.
	m_view.clearCorners();
	m_view.showRectified({});
	m_view.showStatus(tr("Click the 4 corners of the board."));
}

void ManualSetupPresenter::showFrame(const QImage& frame) {
	if (!m_setup.isApplied()) { // A frame queued before apply() is outdated.
		m_view.showImage(frame);
	}
}

void ManualSetupPresenter::showSnapshot(const QImage& warped, const QImage& rectified, const unsigned boardSize) {
	if (!m_setup.isApplied()) { // Analysed just before redo().
		return;
	}

	m_view.showImage(warped);
	m_view.showRectified(rectified);
	m_view.showStatus(boardSize == 0u ? tr("No grid found. Redo to pick the corners again.") : tr("Grid found: %1x%1.").arg(boardSize));
}

} // namespace tengen::vision::gui
