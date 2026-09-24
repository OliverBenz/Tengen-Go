#include "BotDialog.hpp"

#include "GnuGoConfigWidget.hpp"
#include "KataGoConfigWidget.hpp"
#include "Logging.hpp"
#include "model/player.hpp"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QStandardItemModel>
#include <QVBoxLayout>
#include <cassert>
#include <fcntl.h>

namespace tengen::gui {

BotDialog::BotDialog(const engine::InstalledEngines& engines, QWidget* parent)
    : QDialog(parent) {
	setWindowTitle("New Bot Game");

	// Every engine shows, installed or not. One that is not installed cannot be picked, so the default
	// config its widget starts from never leaves the dialog.
	m_engineCombo   = new QComboBox(this);
	m_engineConfigs = new QStackedWidget(this);
	m_gnuGo         = new GnuGoConfigWidget(engines.gnuGo.value_or(engine::GnuGoConfig{}), this);
	m_kataGo        = new KataGoConfigWidget(engines.kataGo.value_or(engine::KataGoConfig{}), this);
	addEngine(tr("GNU Go"), m_gnuGo, engines.gnuGo.has_value());
	addEngine(tr("KataGo"), m_kataGo, engines.kataGo.has_value());
	connect(m_engineCombo, &QComboBox::currentIndexChanged, m_engineConfigs, &QStackedWidget::setCurrentIndex);

	// The combo box starts on its first engine, even when that one cannot be picked.
	const auto* engineItems = qobject_cast<QStandardItemModel*>(m_engineCombo->model());
	for (int row = 0; row < m_engineCombo->count(); ++row) {
		if (engineItems->item(row)->isEnabled()) {
			m_engineCombo->setCurrentIndex(row);
			break;
		}
	}

	m_boardSize = new QComboBox(this);
	m_boardSize->addItem("9x9", 9u);
	m_boardSize->addItem("13x13", 13u);
	m_boardSize->addItem("19x19", 19u);
	m_boardSize->setCurrentIndex(0);

	m_colour = new QComboBox(this);
	m_colour->addItem("Black", static_cast<int>(Player::Black));
	m_colour->addItem("White", static_cast<int>(Player::White));
	m_colour->setCurrentIndex(0);

	// Every engine counts its strength its own way. Its config widget shows which, so the row only says what it sets.
	auto* form = new QFormLayout();
	form->addRow(tr("Engine:"), m_engineCombo);
	form->addRow(tr("Strength:"), m_engineConfigs);
	form->addRow(tr("Board size:"), m_boardSize);
	form->addRow(tr("Your color:"), m_colour);

	// Without an engine there is nothing to play against.
	const bool anyInstalled = engines.gnuGo || engines.kataGo;
	auto* noEngine          = new QLabel(tr("No engine is installed. Please refer to the documentation."), this);
	noEngine->setWordWrap(true);
	noEngine->setVisible(!anyInstalled);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	buttons->button(QDialogButtonBox::Ok)->setEnabled(anyInstalled);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* layout = new QVBoxLayout(this);
	layout->addLayout(form);
	layout->addWidget(noEngine);
	layout->addWidget(buttons);
}

unsigned BotDialog::boardSize() const {
	const unsigned boardSize = m_boardSize->currentData().toUInt();

	if (boardSize != 9 && boardSize != 13 && boardSize != 19) {
		Logger().Log(Logging::LogLevel::Error, "Invalid board size selected in Bot game. Choosing 9x9.");
		return 9u;
	}
	return boardSize;
}

engine::EngineConfig BotDialog::engineConfig() const {
	// Only an installed engine can be picked, and its config widget holds the files the catalog found for it.
	if (m_engineConfigs->currentWidget() == m_kataGo) {
		return m_kataGo->config();
	}
	assert(m_engineConfigs->currentWidget() == m_gnuGo);
	return m_gnuGo->config();
}

bool BotDialog::humanPlaysBlack() const {
	const int player = m_colour->currentData().toInt();

	if (player != static_cast<int>(Player::White) && player != static_cast<int>(Player::Black)) {
		Logger().Log(Logging::LogLevel::Error, "Invalid player selection in Bot game. Choosing Black.");
		return true;
	}
	return static_cast<Player>(player) == Player::Black;
}

void BotDialog::addEngine(const QString& name, QWidget* configWidget, const bool installed) {
	m_engineCombo->addItem(installed ? name : tr("%1 (not installed)").arg(name));
	m_engineConfigs->addWidget(configWidget);
	configWidget->setEnabled(installed);

	if (!installed) {
		// Greyed out, and neither the mouse nor the keyboard can pick it.
		auto* engineItems = qobject_cast<QStandardItemModel*>(m_engineCombo->model());
		assert(engineItems); // A combo box keeps its items in a QStandardItemModel unless it is given another model.
		engineItems->item(m_engineCombo->count() - 1)->setEnabled(false);
	}
}

} // namespace tengen::gui
