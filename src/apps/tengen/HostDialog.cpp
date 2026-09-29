#include "HostDialog.hpp"

#include "BoardSizeWidget.hpp"

#include <QDialogButtonBox>
#include <QFormLayout>
#include <QVBoxLayout>

namespace tengen::gui {

HostDialog::HostDialog(QWidget* parent) : QDialog(parent) {
	setWindowTitle("Host Server");

	m_boardSize = new BoardSizeWidget(this);

	auto* form = new QFormLayout();
	form->addRow(tr("Board size:"), m_boardSize);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addLayout(form);
	mainLayout->addWidget(buttons);
}

unsigned HostDialog::boardSize() const {
	return m_boardSize->boardSize();
}

} // namespace tengen::gui
