#include "vision/gui/manualSetupPresenter.hpp"

#include "vision/gui/manualSetupView.hpp"

#include <QMetaObject>

#include <utility>

namespace tengen::vision::gui {

//! A copy that owns its pixels, so it outlives the image.
static QImage toQImage(const Image& image) {
	return QImage(image.rgb.data(), image.width, image.height, qsizetype{3} * image.width, QImage::Format_RGB888).copy();
}

static int count(const Board& board, const Board::Stone stone) {
	int stones = 0;
	for (unsigned x = 0; x < board.size(); ++x) {
		for (unsigned y = 0; y < board.size(); ++y) {
			stones += board.get({x, y}) == stone ? 1 : 0;
		}
	}
	return stones;
}

//! What the pipeline found, in the colours it marks the stones with.
static QString describe(const Board& board) {
	if (board.size() == 0u) {
		return ManualSetupPresenter::tr("No grid found yet. Redo if the corners are off.");
	}
	return ManualSetupPresenter::tr("%1x%1 grid: %2 black (red), %3 white (blue).").arg(board.size()).arg(count(board, Board::Stone::Black)).arg(count(board, Board::Stone::White));
}

ManualSetupPresenter::ManualSetupPresenter(ManualSetupView& view, const int cameraIndex, const std::filesystem::path& modelPath)
    : m_view(view), m_setup(cameraIndex, modelPath, callbacks()) {
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
	        .onAnalysed   = [onGuiThread, this](const BoardSnapshot& snapshot) { onGuiThread([this, warped = toQImage(snapshot.warped), rectified = toQImage(snapshot.rectified), found = describe(snapshot.board)] { showSnapshot(warped, rectified, found); }); },
	        .onCameraLost = [onGuiThread, this] { onGuiThread([this] { showCameraLost(); }); },
	};
}

void ManualSetupPresenter::onCornerPicked(const QPointF corner) {
	if (m_setup.pick(static_cast<float>(corner.x()), static_cast<float>(corner.y()))) {
		m_view.addCorner(corner);
		showProgress();
	}
}

void ManualSetupPresenter::onApplyRequested() {
	if (!m_setup.apply()) {
		return; // Apply is greyed out until all corners are picked.
	}
	m_view.clearCorners();
	m_view.setApplyAvailable(false, tr("Applied. Redo to pick other corners."));
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
	showProgress();
}

void ManualSetupPresenter::showProgress() {
	const auto picked = m_setup.picked();
	if (picked == ManualSetup::CORNER_COUNT) {
		m_view.setApplyAvailable(true, {});
		m_view.showStatus(tr("Apply to look for the grid, or redo to pick other corners."));
	} else {
		m_view.setApplyAvailable(false, tr("Click all %1 corners first.").arg(ManualSetup::CORNER_COUNT));
		m_view.showStatus(tr("Click the %1 corners of the board, first the one the game shows at the top left: %2 picked.").arg(ManualSetup::CORNER_COUNT).arg(picked));
	}
}

void ManualSetupPresenter::showCameraLost() {
	m_view.setApplyAvailable(false, tr("No camera."));
	m_view.showStatus(tr("No camera, or it stopped delivering."));
}

void ManualSetupPresenter::showFrame(const QImage& frame) {
	if (!m_setup.isApplied()) { // A frame queued before apply() is outdated.
		m_view.showImage(frame);
	}
}

void ManualSetupPresenter::showSnapshot(const QImage& warped, const QImage& rectified, const QString& found) {
	if (!m_setup.isApplied()) { // Analysed just before redo().
		return;
	}

	m_view.showImage(warped);
	m_view.showRectified(rectified);
	m_view.showStatus(found);
}

} // namespace tengen::vision::gui
