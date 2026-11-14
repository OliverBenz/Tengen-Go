#pragma once

#include "boardFile.hpp"
#include "geometryFile.hpp"

#include <QImage>
#include <QObject>

#include <filesystem>
#include <vector>

namespace tengen::labeler {

class LayoutView;

//! Click the stones of one photo seen from above; submit writes them as the board.txt of its folder.
class LayoutPresenter final : public QObject {
	Q_OBJECT

public:
	LayoutPresenter(LayoutView& view, const QImage& photo, const Geometry& geometry, std::filesystem::path file);

private:
	void onPointClicked(int row, int col);
	void onColourChosen(Stone colour);
	void onSubmitRequested();

	void showStones(); //!< The stones so far, the colour a click places, and how many there are of each.

private:
	LayoutView& m_view;
	unsigned m_size;
	std::filesystem::path m_file;
	std::vector<Stone> m_stones;  //!< Row by row.
	Stone m_colour{Stone::black}; //!< What a click places.
};

} // namespace tengen::labeler
