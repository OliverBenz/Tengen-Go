#include "BoardSizeWidget.hpp"

#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPushButton>

namespace tengen::gui {

BoardSizeWidget::BoardSizeWidget(QWidget* parent)
    : QWidget(parent) {
	static constexpr std::array SIZES{9u, 13u, 19u};

	m_sizes = new QButtonGroup(this);
	m_sizes->setExclusive(true);

	// Sits in the dialog's form so it brings no margins of its own.
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(0, 0, 0, 0);

	for (const unsigned size: SIZES) {
		auto* button = new QPushButton(QString("%1x%1").arg(size), this);
		button->setCheckable(true);
		m_sizes->addButton(button, static_cast<int>(size));
		layout->addWidget(button);
	}
	layout->addStretch();

	m_sizes->button(SIZES.front())->setChecked(true);
}

unsigned BoardSizeWidget::boardSize() const {
	return static_cast<unsigned>(m_sizes->checkedId());
}

} // namespace tengen::gui
