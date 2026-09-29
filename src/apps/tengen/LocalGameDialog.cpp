#include "LocalGameDialog.hpp"

#include "Logging.hpp"
#include "RulesConfigWidget.hpp"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QVBoxLayout>

namespace tengen::gui {

LocalGameDialog::LocalGameDialog(QWidget* parent)
    : QDialog(parent) {
	setWindowTitle("New Local Game");

	m_boardSize = new QComboBox(this);
	m_boardSize->addItem("9x9", 9u);
	m_boardSize->addItem("13x13", 13u);
	m_boardSize->addItem("19x19", 19u);
	m_boardSize->setCurrentIndex(0);

	m_rules = new RulesConfigWidget(this);

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

unsigned LocalGameDialog::boardSize() const {
	const unsigned boardSize = m_boardSize->currentData().toUInt();

	if (boardSize != 9 && boardSize != 13 && boardSize != 19) {
		Logger().Log(Logging::LogLevel::Error, "Invalid board size selected in Local game. Choosing 9x9.");
		return 9u;
	}
	return boardSize;
}

GameRules LocalGameDialog::rules() const {
	return m_rules->rules();
}

} // namespace tengen::gui
