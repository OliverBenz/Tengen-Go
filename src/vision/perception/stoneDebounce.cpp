#include "stoneDebounce.hpp"

#include <cstddef>

namespace tengen::vision {

static constexpr unsigned STABLE_FRAMES = 5u; //!< Frames in a row a state must be found before it counts.

const std::vector<core::StoneState>& StoneDebounce::update(const std::vector<core::StoneState>& found) {
	if (found.size() != m_steady.size()) {
		m_steady.assign(found.size(), core::StoneState::Empty);
		m_candidate = m_steady;
		m_streak.assign(found.size(), 0u);
	}

	for (std::size_t i = 0; i < found.size(); ++i) {
		m_streak[i]    = found[i] == m_candidate[i] ? m_streak[i] + 1u : 1u;
		m_candidate[i] = found[i];
		if (m_streak[i] >= STABLE_FRAMES) {
			m_steady[i] = found[i];
		}
	}
	return m_steady;
}

void StoneDebounce::reset() {
	*this = {};
}

} // namespace tengen::vision
