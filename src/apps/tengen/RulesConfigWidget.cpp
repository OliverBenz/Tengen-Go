#include "RulesConfigWidget.hpp"

#include "gui/resources.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QVBoxLayout>
#include <cmath>

namespace tengen::gui {
namespace {

constexpr int customRuleSet = -1;    //!< Data of the custom entry in the ruleset selector. No RuleSet has this value.
constexpr double komiLimit  = 100.0; //!< Largest komi either way. Negative komi goes to black instead.

} // namespace

RulesConfigWidget::RulesConfigWidget(QWidget* parent)
    : QWidget(parent) {
	m_ruleSet = new QComboBox(this);
	for (const auto ruleSet: {RuleSet::Japanese, RuleSet::Chinese, RuleSet::Korean}) {
		m_ruleSet->addItem(gameRules::displayName(ruleSet), static_cast<int>(ruleSet));
	}
	m_ruleSet->addItem(tr("Custom"), customRuleSet);
	m_ruleSet->setCurrentIndex(0);

	m_scoring = new QComboBox(this);
	for (const auto scoring: {Scoring::Territory, Scoring::Area}) {
		m_scoring->addItem(gameRules::displayName(scoring), static_cast<int>(scoring));
	}

	m_ko = new QComboBox(this);
	for (const auto ko: {Ko::Simple, Ko::Situational, Ko::Positional}) {
		m_ko->addItem(gameRules::displayName(ko), static_cast<int>(ko));
	}

	m_komi = new QDoubleSpinBox(this);
	m_komi->setRange(-komiLimit, komiLimit);
	m_komi->setDecimals(1);
	m_komi->setSingleStep(0.5);

	// Round komi to half points after editing.
	connect(m_komi, &QDoubleSpinBox::editingFinished, this, [this] { m_komi->setValue(std::round(m_komi->value() * 2.0) / 2.0); });

	m_suicide = new QCheckBox(tr("Allowed"), this);

	// Default Custom values are Japanese rules
	const GameRules defaults = fromRuleSet(RuleSet::Japanese);
	m_scoring->setCurrentIndex(m_scoring->findData(static_cast<int>(defaults.scoringMethod)));
	m_ko->setCurrentIndex(m_ko->findData(static_cast<int>(defaults.koRule)));
	m_komi->setValue(defaults.komi);
	m_suicide->setChecked(defaults.suicideLegal);

	m_custom         = new QWidget(this);
	auto* customForm = new QFormLayout(m_custom);
	customForm->setContentsMargins(0, 0, 0, 0);
	customForm->addRow(tr("Scoring:"), m_scoring);
	customForm->addRow(tr("Ko:"), m_ko);
	customForm->addRow(tr("Komi:"), m_komi);
	customForm->addRow(tr("Suicide:"), m_suicide);
	m_custom->hide();

	// Keep custom field values while hidden so we can switch between without losing config.
	connect(m_ruleSet, &QComboBox::currentIndexChanged, this, [this] {
		m_custom->setVisible(m_ruleSet->currentData().toInt() == customRuleSet);
	});

	// Sits in the dialog's form like any other field, so it brings no margins of its own.
	auto* layout = new QVBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(m_ruleSet);
	layout->addWidget(m_custom);
}

GameRules RulesConfigWidget::rules() const {
	// No range checks: the selectors only hold the values they were given.
	const int ruleSet = m_ruleSet->currentData().toInt();
	if (ruleSet != customRuleSet) {
		return fromRuleSet(static_cast<RuleSet>(ruleSet));
	}

	return {
	        .scoringMethod = static_cast<Scoring>(m_scoring->currentData().toInt()),
	        .koRule        = static_cast<Ko>(m_ko->currentData().toInt()),
	        .komi          = static_cast<float>(m_komi->value()),
	        .suicideLegal  = m_suicide->isChecked(),
	};
}

} // namespace tengen::gui
