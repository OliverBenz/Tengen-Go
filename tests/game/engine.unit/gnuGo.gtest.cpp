#include "engine/gnuGo.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace tengen::gtest {

using Arguments = std::vector<std::string>;

TEST(GnuGo, StandardRuleSets) {
	EXPECT_EQ(engine::GnuGo::ruleArguments(fromRuleSet(RuleSet::Japanese)), (Arguments{"--japanese-rules", "--simple-ko", "--forbid-suicide"}));
	EXPECT_EQ(engine::GnuGo::ruleArguments(fromRuleSet(RuleSet::Chinese)), (Arguments{"--chinese-rules", "--positional-superko", "--forbid-suicide"}));
	EXPECT_EQ(engine::GnuGo::ruleArguments(fromRuleSet(RuleSet::Korean)), (Arguments{"--japanese-rules", "--simple-ko", "--forbid-suicide"}));
}

TEST(GnuGo, CustomRules) {
	const GameRules rules{.scoringMethod = Scoring::Area, .koRule = Ko::Situational, .komi = 0.5f, .suicideLegal = true};

	EXPECT_EQ(engine::GnuGo::ruleArguments(rules), (Arguments{"--chinese-rules", "--situational-superko", "--allow-suicide"}));
}

} // namespace tengen::gtest
