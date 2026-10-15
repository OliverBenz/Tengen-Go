#include "LocalGameDialog.hpp"

#include "BoardSizeWidget.hpp"
#include "RulesConfigWidget.hpp"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QVBoxLayout>

namespace tengen::gui {

LocalGameDialog::LocalGameDialog(QWidget* parent) : QDialog(parent) {
	setWindowTitle("New Local Game");

	m_boardSize = new BoardSizeWidget(this);
	m_rules     = new RulesConfigWidget(this);

	auto* form = new QFormLayout();
	form->addRow(tr("Board size:"), m_boardSize);
	form->addRow(tr("Rules:"), m_rules);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* layout = new QVBoxLayout(this);
	layout->setSizeConstraint(QLayout::SetFixedSize); // Shrink back once the custom rules hide again.
	layout->addLayout(form);
	layout->addWidget(buttons);
}

GameConfig LocalGameDialog::config() const {
	return GameConfig{.boardSize = m_boardSize->boardSize(), .rules = m_rules->rules()};
}

} // namespace tengen::gui
