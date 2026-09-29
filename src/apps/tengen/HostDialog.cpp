#include "HostDialog.hpp"

#include "BoardSizeWidget.hpp"
#include "RulesConfigWidget.hpp"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QVBoxLayout>

namespace tengen::gui {

HostDialog::HostDialog(QWidget* parent) : QDialog(parent) {
	setWindowTitle("Host Server");

	m_boardSize = new BoardSizeWidget(this);
	m_rules     = new RulesConfigWidget(this);

	auto* form = new QFormLayout();
	form->addRow(tr("Board size:"), m_boardSize);
	form->addRow(tr("Rules:"), m_rules);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->setSizeConstraint(QLayout::SetFixedSize); // Shrink back once the custom rules hide again.
	mainLayout->addLayout(form);
	mainLayout->addWidget(buttons);
}

unsigned HostDialog::boardSize() const {
	return m_boardSize->boardSize();
}

GameRules HostDialog::rules() const {
	return m_rules->rules();
}

} // namespace tengen::gui
