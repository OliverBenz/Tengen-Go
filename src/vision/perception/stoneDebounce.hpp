#pragma once

#include "vision/core/stoneFinder.hpp"

#include <vector>

namespace tengen::vision {

//! Steadies the stones found frame by frame: an intersection only changes once it was found the same for several frames in a row.
//! Glare can show a false stone for a frame or two, see models/stoneNet.json.
class StoneDebounce {
public:
	//! \param [in] found Stones of the latest frame. Another board size starts over on an empty board.
	//! \returns    The steady state of every intersection.
	const std::vector<core::StoneState>& update(const std::vector<core::StoneState>& found);
	void reset(); //!< Forget the board, e.g. before watching another one.

private:
	std::vector<core::StoneState> m_steady;    //!< What update() reports.
	std::vector<core::StoneState> m_candidate; //!< The latest frame.
	std::vector<unsigned> m_streak;            //!< Frames in a row m_candidate[i] was found.
};

} // namespace tengen::vision
