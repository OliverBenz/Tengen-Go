#include "HostDialog.hpp"

#include "BoardSizeWidget.hpp"
#include "PlayerColourWidget.hpp"
#include "RulesConfigWidget.hpp"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QVBoxLayout>

namespace tengen::gui {

HostDialog::HostDialog(QWidget* parent) : QDialog(parent) {
	setWindowTitle("Host Server");

	m_boardSize = new BoardSizeWidget(this);
	m_colour    = new PlayerColourWidget(this);
	m_rules     = new RulesConfigWidget(this);

	auto* form = new QFormLayout();
	form->addRow(tr("Board size:"), m_boardSize);
	form->addRow(tr("Your color:"), m_colour);
	form->addRow(tr("Rules:"), m_rules); // Last, so the custom rules unfold below everything else.

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->setSizeConstraint(QLayout::SetFixedSize); // Shrink back once the custom rules hide again.
	mainLayout->addLayout(form);
	mainLayout->addWidget(buttons);
}

GameConfig HostDialog::config() const {
	return GameConfig{.boardSize = m_boardSize->boardSize(), .rules = m_rules->rules()};
}

Player HostDialog::hostColour() const {
	return m_colour->player();
}

} // namespace tengen::gui
