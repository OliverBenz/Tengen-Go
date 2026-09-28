#include "core/gameRules.hpp"

#include <cassert>

namespace tengen {

GameRules fromRuleSet(RuleSet ruleSet) {
	switch (ruleSet) {
	case RuleSet::Japanese:
		return {
		        .scoringMethod = Scoring::Territory,
		        .koRule        = Ko::Simple,
		        .komi          = 6.5,
		        .suicideLegal  = false,
		};
	case RuleSet::Chinese:
		return {
		        .scoringMethod = Scoring::Area,
		        .koRule        = Ko::Positional,
		        .komi          = 7.5,
		        .suicideLegal  = false,
		};
	case RuleSet::Korean:
		return {
		        .scoringMethod = Scoring::Territory,
		        .koRule        = Ko::Simple,
		        .komi          = 6.5,
		        .suicideLegal  = false,
		};
	}

	assert(false);
	return fromRuleSet(RuleSet::Japanese);
}

} // namespace tengen
