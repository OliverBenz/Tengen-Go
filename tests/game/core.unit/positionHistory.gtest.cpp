#include "core/positionHistory.hpp"

#include <gtest/gtest.h>

namespace tengen::gtest {

// Board hashes are arbitrary numbers here. Only equality matters to the history.
static constexpr uint64_t START     = 1u;
static constexpr uint64_t KO_TAKEN  = 2u;
static constexpr uint64_t ELSEWHERE = 3u;

TEST(PositionHistory, SimpleKo) {
	PositionHistory history(Ko::Simple);
	history.record(START, Player::Black);
	history.record(KO_TAKEN, Player::White);

	EXPECT_FALSE(history.allows(START, Player::Black));

	history.record(ELSEWHERE, Player::Black);
	EXPECT_TRUE(history.allows(START, Player::White));
	EXPECT_FALSE(history.allows(KO_TAKEN, Player::White));
}

TEST(PositionHistory, SituationalKo) {
	PositionHistory history(Ko::Situational);
	history.record(START, Player::Black);
	history.record(KO_TAKEN, Player::White);
	history.record(ELSEWHERE, Player::Black);

	EXPECT_FALSE(history.allows(START, Player::Black));
	EXPECT_TRUE(history.allows(START, Player::White));
	EXPECT_FALSE(history.allows(KO_TAKEN, Player::White));
	EXPECT_TRUE(history.allows(KO_TAKEN, Player::Black));
}

TEST(PositionHistory, PositionalKo) {
	PositionHistory history(Ko::Positional);
	history.record(START, Player::Black);
	history.record(KO_TAKEN, Player::White);
	history.record(ELSEWHERE, Player::Black);

	EXPECT_FALSE(history.allows(START, Player::Black));
	EXPECT_FALSE(history.allows(START, Player::White));
	EXPECT_FALSE(history.allows(KO_TAKEN, Player::Black));
	EXPECT_TRUE(history.allows(4u, Player::White));
}

} // namespace tengen::gtest
