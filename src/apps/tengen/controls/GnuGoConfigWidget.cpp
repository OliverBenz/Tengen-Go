#include "GnuGoConfigWidget.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <utility>

namespace tengen::gui {

static QString levelText(const int level) {
	return GnuGoConfigWidget::tr("Level %1").arg(level);
}

GnuGoConfigWidget::GnuGoConfigWidget(engine::GnuGoConfig config, QWidget* parent)
    : QWidget(parent), m_config(std::move(config)) {
	// The slider runs over GNU Go's own levels, so it offers every one of them.
	m_level = new QSlider(Qt::Horizontal, this);
	m_level->setRange(engine::GnuGoConfig::weakestLevel, engine::GnuGoConfig::strongestLevel);
	m_level->setValue(m_config.level);
	m_level->setTickPosition(QSlider::TicksBelow);
	m_level->setTickInterval(1);
	m_level->setPageStep(1);

	m_levelLabel = new QLabel(levelText(m_level->value()), this);
	m_levelLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	// Hold the width of the longest level so the slider does not shift while it is dragged.
	m_levelLabel->setMinimumWidth(m_levelLabel->fontMetrics().horizontalAdvance(levelText(engine::GnuGoConfig::strongestLevel)));

	connect(m_level, &QSlider::valueChanged, this, [this](const int level) {
		m_levelLabel->setText(levelText(level));
	});

	// Sits in the dialog's form like any other field, so it brings no margins of its own.
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(m_level);
	layout->addWidget(m_levelLabel);
}

engine::GnuGoConfig GnuGoConfigWidget::config() const {
	engine::GnuGoConfig picked = m_config;
	picked.level               = m_level->value(); // No range check: the slider cannot leave the range it was given.
	return picked;
}

} // namespace tengen::gui
