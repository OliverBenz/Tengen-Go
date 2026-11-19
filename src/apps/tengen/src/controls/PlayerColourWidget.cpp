#include "PlayerColourWidget.hpp"

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPushButton>

namespace tengen::gui {

PlayerColourWidget::PlayerColourWidget(QWidget* parent)
    : QWidget(parent) {
	m_colours = new QButtonGroup(this);
	m_colours->setExclusive(true);

	// Sits in the dialog's form so it brings no margins of its own.
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);

	const auto addColour = [&](const QString& name, const Player player) {
		auto* button = new QPushButton(name, this);
		button->setCheckable(true);
		m_colours->addButton(button, static_cast<int>(player));
		layout->addWidget(button);
	};
	addColour(tr("Black"), Player::Black);
	addColour(tr("White"), Player::White);
	layout->addStretch();

	m_colours->button(static_cast<int>(Player::Black))->setChecked(true);
}

Player PlayerColourWidget::player() const {
	return static_cast<Player>(m_colours->checkedId());
}

} // namespace tengen::gui
