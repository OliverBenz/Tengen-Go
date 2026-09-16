#pragma once

#include "corpus.hpp"

namespace tengen::vision::analysis {

//! Which labelled outline BoardFinder's quad landed on. Both are valid (see boardFinder.hpp).
enum class Framing {
	None,
	Board,
	Grid
};

//! Geometry measured for one image. Fields stay at their defaults when an earlier stage failed.
struct Measurement {
	std::filesystem::path image;
	bool boardFound{};    //!< BoardFinder produced a valid warp.
	Corners contour{};    //!< BoardFinder's quad, original image space.
	unsigned boardSize{}; //!< GridFinder's detected size, 0 when it failed.
	double spacing{};     //!< GridFinder's grid spacing in B space (px).

	Framing framing{Framing::None};
	double errorToBoard{}; //!< Worst corner distance to the labelled board outline (px).
	double errorToGrid{};  //!< Worst corner distance to the labelled grid outline (px).
	double gridFill{};     //!< Fraction of the B_0 canvas spanned by the labelled grid.
	double spacingError{}; //!< Spacing deviation from the labelled grid, as a signed fraction.
};

Measurement measureImage(const std::filesystem::path& image, const std::optional<GroundTruth>& truth);

//! Worst corner distance under the permutation minimising total distance, since corners are unordered.
double cornerDistance(const Corners& lhs, const Corners& rhs);

//! Per-corner median quad. A fixed-camera series should hold still, so this is its reference.
Corners medianQuad(const std::vector<Corners>& quads);

} // namespace tengen::vision::analysis
