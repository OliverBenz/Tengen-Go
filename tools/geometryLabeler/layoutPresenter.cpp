#include "layoutPresenter.hpp"

#include "layoutView.hpp"
#include "rectify.hpp"

#include <algorithm>
#include <utility>

namespace tengen::labeler {

static constexpr int SPACING = 40; //!< [px] between the grid lines of the board seen from above.

LayoutPresenter::LayoutPresenter(LayoutView& view, const QImage& photo, const Geometry& geometry, std::filesystem::path file)
    : m_view(view), m_size(geometry.boardSize), m_file(std::move(file)), m_stones(std::size_t{m_size} * m_size, Stone::empty) {
	connect(&m_view, &LayoutView::pointClicked, this, &LayoutPresenter::onPointClicked);
	connect(&m_view, &LayoutView::colourChosen, this, &LayoutPresenter::onColourChosen);
	connect(&m_view, &LayoutView::submitRequested, this, &LayoutPresenter::onSubmitRequested);
	m_view.showBoard(rectify(photo, geometry, SPACING), m_size);
	showStones();
}

void LayoutPresenter::onPointClicked(const int row, const int col) {
	Stone& stone = m_stones[static_cast<std::size_t>(row) * m_size + static_cast<std::size_t>(col)];
	stone        = stone == m_colour ? Stone::empty : m_colour;
	showStones();
}

void LayoutPresenter::onColourChosen(const Stone colour) {
	m_colour = colour;
	showStones();
}

void LayoutPresenter::onSubmitRequested() {
	const QString name = QString::fromStdU16String(m_file.u16string());
	if (std::filesystem::exists(m_file)) {
		m_view.showStatus(tr("%1 already exists.").arg(name));
		return;
	}
	if (!writeBoard(m_file, m_size, m_stones)) {
		m_view.showStatus(tr("Cannot write %1.").arg(name));
		return;
	}
	m_view.accept();
}

void LayoutPresenter::showStones() {
	m_view.showStones(m_stones);
	const auto count = [this](const Stone stone) { return QString::number(std::ranges::count(m_stones, stone)); };
	m_view.showStatus(tr("A click places %1 (B / W to switch), a second click removes it. %2 black, %3 white.")
	                          .arg(m_colour == Stone::black ? tr("black") : tr("white"), count(Stone::black), count(Stone::white)));
}

} // namespace tengen::labeler
