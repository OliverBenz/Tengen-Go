#include "BoardSizeWidget.hpp"

#include "model/gameConfig.hpp"

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPushButton>

namespace tengen::gui {

BoardSizeWidget::BoardSizeWidget(QWidget* parent)
    : QWidget(parent) {
	m_sizes = new QButtonGroup(this);
	m_sizes->setExclusive(true);

	// Sits in the dialog's form so it brings no margins of its own.
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);

	for (const auto size: SUPPORTED_BOARD_SIZES) {
		auto* button = new QPushButton(QString("%1x%1").arg(size), this);
		button->setCheckable(true);
		m_sizes->addButton(button, static_cast<int>(size));
		layout->addWidget(button);
	}
	layout->addStretch();

	m_sizes->button(static_cast<int>(SUPPORTED_BOARD_SIZES.front()))->setChecked(true);
}

unsigned BoardSizeWidget::boardSize() const {
	return static_cast<unsigned>(m_sizes->checkedId());
}

} // namespace tengen::gui
