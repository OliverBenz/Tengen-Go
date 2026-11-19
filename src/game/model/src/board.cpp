#include "model/board.hpp"

#include <cassert>

namespace tengen {

Board::Board(unsigned size)
    : m_size(size), m_board(static_cast<std::size_t>(size) * size, Stone::Empty) {
}

unsigned Board::size() const {
	return m_size;
}

bool Board::place(Coord c, Stone value) {
	assert(c.x < m_size && c.y < m_size); // Game should verify valid coordinate.
	assert(value != Stone::Empty);        // Use remove

	if (isEmpty(c)) {
		m_board[c.y * m_size + c.x] = value;
		return true;
	}
	return false;
}

bool Board::remove(Coord c) {
	assert(c.x < m_size && c.y < m_size); // Game should verify valid coordinate.

	if (!isEmpty(c)) {
		m_board[c.y * m_size + c.x] = Stone::Empty;
		return true;
	}
	return false;
}

Board::Stone Board::get(Coord c) const {
	assert(c.x < m_size && c.y < m_size); // Game should verify valid coordinate.
	return m_board[c.y * m_size + c.x];
}

bool Board::isEmpty(Coord c) const {
	assert(c.x < m_size && c.y < m_size); // Game should verify valid coordinate.
	return get(c) == Stone::Empty;
}

} // namespace tengen
