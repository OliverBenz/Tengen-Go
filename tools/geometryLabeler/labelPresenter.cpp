#include "labelPresenter.hpp"

#include "boardFile.hpp"
#include "geometryFile.hpp"
#include "labelView.hpp"

#include <QImage>
#include <QImageReader>

#include <algorithm>
#include <optional>
#include <utility>

namespace tengen::labeler {

static constexpr std::size_t CORNERS = 4; //!< Per outline: the board's, then the grid's.

static QString toQString(const std::filesystem::path& path) {
	return QString::fromStdU16String(path.u16string());
}

//! Upright by its EXIF orientation, like cv::imread, so the coordinates match. Null if it cannot be read.
static QImage readPhoto(const std::filesystem::path& photo) {
	QImageReader reader(toQString(photo));
	reader.setAutoTransform(true);
	return reader.read();
}

static Pixel toPixel(const QPoint point) {
	return {point.x(), point.y()};
}

LabelPresenter::LabelPresenter(LabelView& view, LayoutRunner runLayout)
    : m_view(view), m_runLayout(std::move(runLayout)) {
	connect(&m_view, &LabelView::folderChosen, this, &LabelPresenter::onFolderChosen);
	connect(&m_view, &LabelView::cornerPicked, this, &LabelPresenter::onCornerPicked);
	connect(&m_view, &LabelView::undoRequested, this, &LabelPresenter::onUndoRequested);
	connect(&m_view, &LabelView::moveRequested, this, &LabelPresenter::onMoveRequested);
	connect(&m_view, &LabelView::skipRequested, this, &LabelPresenter::onSkipRequested);
	connect(&m_view, &LabelView::boardSizeChosen, this, &LabelPresenter::onBoardSizeChosen);
	connect(&m_view, &LabelView::submitRequested, this, &LabelPresenter::onSubmitRequested);
	connect(&m_view, &LabelView::layoutRequested, this, &LabelPresenter::onLayoutRequested);
	m_view.showBoardSize(m_boardSize);
	m_view.showStatus(tr("Open a folder with photos."));
	showLayoutAvailability();
}

void LabelPresenter::onFolderChosen(const QString& folder) {
	m_folder  = std::filesystem::path(folder.toStdU16String());
	m_photos  = unlabelledPhotos(m_folder);
	m_current = 0;
	showPhoto();
	showLayoutAvailability();
}

void LabelPresenter::onCornerPicked(const QPoint pixel) {
	if (m_corners.size() < 2 * CORNERS) {
		m_corners.push_back(pixel);
		showCorners();
	}
}

void LabelPresenter::onUndoRequested() {
	if (!m_corners.empty()) {
		m_corners.pop_back();
		showCorners();
	}
}

void LabelPresenter::onMoveRequested(const QPoint step) {
	if (!m_corners.empty()) {
		m_corners.back() += step;
		showCorners();
	}
}

void LabelPresenter::onSkipRequested() {
	if (hasPhoto()) {
		++m_current;
		showPhoto();
	}
}

void LabelPresenter::onBoardSizeChosen(const unsigned size) {
	m_boardSize = size;
	m_view.showBoardSize(m_boardSize);
}

void LabelPresenter::onSubmitRequested() {
	if (!hasPhoto()) {
		return;
	}
	if (m_corners.size() < 2 * CORNERS) {
		m_view.showStatus(tr("Click all 8 corners first."));
		return;
	}

	Geometry geometry{.boardSize = m_boardSize};
	for (std::size_t i = 0; i < CORNERS; ++i) {
		geometry.boardCorners[i] = toPixel(m_corners[i]);
		geometry.gridCorners[i]  = toPixel(m_corners[CORNERS + i]);
	}
	const std::filesystem::path file = geometryFile(m_photos[m_current]);
	if (!writeGeometry(file, geometry)) {
		m_view.showStatus(tr("Cannot write %1.").arg(toQString(file)));
		return;
	}
	++m_current;
	showPhoto();
	showLayoutAvailability();
}

void LabelPresenter::onLayoutRequested() {
	const QString chosen = m_view.choosePhoto(toQString(m_folder));
	if (chosen.isEmpty()) {
		return;
	}
	const std::filesystem::path photo(chosen.toStdU16String());
	const std::optional<Geometry> geometry = readGeometry(geometryFile(photo));
	const QImage image                     = readPhoto(photo);
	if (!geometry || image.isNull()) {
		m_view.showStatus(tr("Cannot read %1 or its geometry file.").arg(toQString(photo.filename())));
		return;
	}

	if (m_runLayout(image, *geometry, boardFile(m_folder))) {
		m_view.showStatus(tr("Wrote %1.").arg(toQString(boardFile(m_folder))));
	}
	showLayoutAvailability();
}

void LabelPresenter::showPhoto() {
	m_corners.clear();
	if (!hasPhoto()) {
		m_view.showPhoto({});
		m_view.showCorners({}, {});
		m_view.showStatus(tr("No photos left to label in this folder."));
		return;
	}

	const QImage photo = readPhoto(m_photos[m_current]);
	m_view.showPhoto(photo);
	showCorners();
	if (photo.isNull()) {
		m_view.showStatus(tr("Cannot read %1.").arg(toQString(m_photos[m_current])));
	}
}

void LabelPresenter::showCorners() {
	const auto split = static_cast<std::ptrdiff_t>(std::min(m_corners.size(), CORNERS));
	m_view.showCorners({m_corners.begin(), m_corners.begin() + split}, {m_corners.begin() + split, m_corners.end()});
	const QString photo = toQString(m_photos[m_current].filename());
	m_view.showStatus(tr("%1 (%2 of %3): %4").arg(photo, QString::number(m_current + 1), QString::number(m_photos.size()), nextStep()));
}

bool LabelPresenter::hasPhoto() const {
	return m_current < m_photos.size();
}

QString LabelPresenter::nextStep() const {
	if (m_corners.size() < CORNERS) {
		return tr("click board corner %1 of 4, going around the board's outer edge.").arg(m_corners.size() + 1);
	}
	if (m_corners.size() < 2 * CORNERS) {
		return tr("click grid corner %1 of 4, going around the outermost lines.").arg(m_corners.size() - CORNERS + 1);
	}
	return tr("submit, or undo the last corner.");
}

void LabelPresenter::showLayoutAvailability() {
	QString blocker;
	if (m_folder.empty()) {
		blocker = tr("Open a folder first.");
	} else if (std::filesystem::exists(boardFile(m_folder))) {
		blocker = tr("board.txt already exists.");
	} else if (!unlabelledPhotos(m_folder).empty()) {
		blocker = tr("Every photo needs its geometry file first.");
	}
	m_view.showLayoutAvailable(blocker.isEmpty(), blocker.isEmpty() ? tr("Click the stones of one photo to write board.txt.") : blocker);
}

} // namespace tengen::labeler
