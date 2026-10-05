#include "engine/kataGo.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <string>

namespace tengen::gtest {

TEST(KataGo, StandardRuleSets) {
	EXPECT_EQ(engine::KataGo::setRulesCommand(fromRuleSet(RuleSet::Japanese)),
	          R"(kata-set-rules {"ko":"SIMPLE","scoring":"TERRITORY","tax":"SEKI","suicide":false,"friendlyPassOk":false})");
	EXPECT_EQ(engine::KataGo::setRulesCommand(fromRuleSet(RuleSet::Chinese)),
	          R"(kata-set-rules {"ko":"POSITIONAL","scoring":"AREA","tax":"NONE","suicide":false,"friendlyPassOk":true})");
}

TEST(KataGo, CustomRules) {
	const GameRules rules{.scoringMethod = Scoring::Territory, .koRule = Ko::Situational, .komi = 0.5f, .suicideLegal = true};

	EXPECT_EQ(engine::KataGo::setRulesCommand(rules),
	          R"(kata-set-rules {"ko":"SITUATIONAL","scoring":"TERRITORY","tax":"SEKI","suicide":true,"friendlyPassOk":false})");
}

TEST(KataGo, AcceptsKomi) {
	EXPECT_TRUE(engine::KataGo::acceptsKomi(0.0f));
	EXPECT_TRUE(engine::KataGo::acceptsKomi(6.5f));
	EXPECT_TRUE(engine::KataGo::acceptsKomi(7.0f));
	EXPECT_TRUE(engine::KataGo::acceptsKomi(-0.5f)); // Black receives komi.

	EXPECT_FALSE(engine::KataGo::acceptsKomi(6.3f));
	EXPECT_FALSE(engine::KataGo::acceptsKomi(6.75f));
	EXPECT_FALSE(engine::KataGo::acceptsKomi(std::numeric_limits<float>::quiet_NaN()));
	EXPECT_FALSE(engine::KataGo::acceptsKomi(std::numeric_limits<float>::infinity()));
}

TEST(KataGo, RulesCommandIsOneArgument) {
	// GTP separates arguments by whitespace, so the rules must not contain any.
	const std::string command = engine::KataGo::setRulesCommand(fromRuleSet(RuleSet::Chinese));
	EXPECT_EQ(command.find(' '), std::string{"kata-set-rules"}.size());
	EXPECT_EQ(command.find(' ', command.find(' ') + 1), std::string::npos);
}

} // namespace tengen::gtest
