#pragma once

#include "geometryFile.hpp"

#include <QImage>
#include <QObject>
#include <QPoint>
#include <QString>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <vector>

namespace tengen::labeler {

class LabelView;

//! Lets the user click the stones of `photo` and write them as `boardFile`. True if the file was written.
using LayoutRunner = std::function<bool(const QImage& photo, const Geometry& geometry, const std::filesystem::path& boardFile)>;

//! Walks through the photos of a folder that have no geometry file yet: click the board's 4 corners, then the grid's,
//! and submit writes the file and moves on to the next photo. Once every photo has one, the folder's board.txt can
//! be written from one photo, see LayoutPresenter, which `runLayout` drives.
class LabelPresenter final : public QObject {
	Q_OBJECT

public:
	LabelPresenter(LabelView& view, LayoutRunner runLayout);

private:
	void onFolderChosen(const QString& folder);
	void onCornerPicked(QPoint pixel);
	void onUndoRequested();
	void onMoveRequested(QPoint step);
	void onSkipRequested();
	void onBoardSizeChosen(unsigned size);
	void onSubmitRequested();
	void onLayoutRequested();

	void showPhoto();      //!< The current photo with no corners yet, or that the folder is done.
	void showCorners();    //!< The corners so far and which one to click next.
	bool hasPhoto() const; //!< False before a folder is chosen and after its last photo.
	QString nextStep() const;
	void showLayoutAvailability(); //!< Only for a folder whose photos all have a geometry file and no board.txt yet.

private:
	LabelView& m_view;
	LayoutRunner m_runLayout;
	std::filesystem::path m_folder{};
	std::vector<std::filesystem::path> m_photos{};
	std::size_t m_current{0};
	unsigned m_boardSize{19};
	std::vector<QPoint> m_corners{}; //!< The board's corners first, then the grid's.
};

} // namespace tengen::labeler
