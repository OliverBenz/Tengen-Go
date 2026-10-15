#include "KataGoConfigWidget.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <utility>

namespace tengen::gui {
namespace {

QString rankText(const Skill rank) {
	return QString::fromStdString(toString(rank));
}

} // namespace

KataGoConfigWidget::KataGoConfigWidget(engine::KataGoConfig config, QWidget* parent)
    : QWidget(parent), m_config(std::move(config)) {
	// The slider runs over the skill scale itself, so every rank in between is offered too.
	m_rank = new QSlider(Qt::Horizontal, this);
	m_rank->setRange(static_cast<int>(engine::KataGoConfig::weakestRank), static_cast<int>(engine::KataGoConfig::strongestRank));
	m_rank->setValue(static_cast<int>(m_config.rank));
	m_rank->setTickPosition(QSlider::TicksBelow);
	m_rank->setTickInterval(1);
	m_rank->setPageStep(1);

	m_rankLabel = new QLabel(rankText(Skill{static_cast<int8_t>(m_rank->value())}), this);
	m_rankLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	// Hold the width of the longest rank so the slider does not shift while it is dragged.
	m_rankLabel->setMinimumWidth(m_rankLabel->fontMetrics().horizontalAdvance("30k"));

	connect(m_rank, &QSlider::valueChanged, this, [this](const int value) {
		m_rankLabel->setText(rankText(Skill{static_cast<int8_t>(value)}));
	});

	// Sits in the dialog's form like any other field, so it brings no margins of its own.
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);
	layout->addWidget(m_rank);
	layout->addWidget(m_rankLabel);
}

engine::KataGoConfig KataGoConfigWidget::config() const {
	engine::KataGoConfig picked = m_config;
	picked.rank                 = Skill{static_cast<int8_t>(m_rank->value())}; // No range check: the slider cannot leave the range it was given.
	return picked;
}

} // namespace tengen::gui
