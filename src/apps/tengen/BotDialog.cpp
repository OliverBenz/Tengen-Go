#include "BotDialog.hpp"

#include "Logging.hpp"
#include "model/player.hpp"

#include <QComboBox>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSlider>
#include <QVBoxLayout>
#include <fcntl.h>

namespace tengen::gui {
namespace {

// The bot imitates human ranks, and only the ones it was trained on: 20k is the weakest it knows.
// Anything below that is a matter of handicap stones, which the game does not offer yet.
constexpr Skill weakestBot = fromKyu(20);

// The imitation stops matching the rank somewhere in the low dan ranks, where playing that strongly
// takes search rather than imitation alone. Offering ranks we cannot honestly play would be a lie.
constexpr Skill strongestBot = fromDan(3);

QString rankText(const Skill skill) {
	return QString::fromStdString(toString(skill));
}

} // namespace

BotDialog::BotDialog(QWidget* parent)
    : QDialog(parent) {
	setWindowTitle("New Bot Game");

	m_boardSize = new QComboBox(this);
	m_boardSize->addItem("9x9", 9u);
	m_boardSize->addItem("13x13", 13u);
	m_boardSize->addItem("19x19", 19u);
	m_boardSize->setCurrentIndex(0);

	// The slider runs over the skill scale itself, so every rank in between is offered too.
	m_skill = new QSlider(Qt::Horizontal, this);
	m_skill->setRange(static_cast<int>(weakestBot), static_cast<int>(strongestBot));
	m_skill->setValue(static_cast<int>(weakestBot));
	m_skill->setTickPosition(QSlider::TicksBelow);
	m_skill->setTickInterval(1);
	m_skill->setPageStep(1);

	m_skillLabel = new QLabel(rankText(weakestBot), this);
	m_skillLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
	// Hold the width of the longest rank so the slider does not shift while it is dragged.
	m_skillLabel->setMinimumWidth(m_skillLabel->fontMetrics().horizontalAdvance("30k"));

	connect(m_skill, &QSlider::valueChanged, this, [this](const int value) {
		m_skillLabel->setText(rankText(Skill{static_cast<int8_t>(value)}));
	});

	auto* skillRow = new QHBoxLayout();
	skillRow->addWidget(m_skill);
	skillRow->addWidget(m_skillLabel);

	m_colour = new QComboBox(this);
	m_colour->addItem("Black", static_cast<int>(Player::Black));
	m_colour->addItem("White", static_cast<int>(Player::White));
	m_colour->setCurrentIndex(0);

	auto* form = new QFormLayout();
	form->addRow(tr("Board size:"), m_boardSize);
	form->addRow(tr("Opponent rank:"), skillRow);
	form->addRow(tr("Your color:"), m_colour);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* layout = new QVBoxLayout(this);
	layout->addLayout(form);
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

Skill BotDialog::skill() const {
	// No range check: the slider cannot leave the range it was given, unlike a combo box's user data.
	return Skill{static_cast<int8_t>(m_skill->value())};
}

bool BotDialog::humanPlaysBlack() const {
	const int player = m_colour->currentData().toInt();

	if (player != static_cast<int>(Player::White) && player != static_cast<int>(Player::Black)) {
		Logger().Log(Logging::LogLevel::Error, "Invalid player selection in Bot game. Choosing Black.");
		return true;
	}
	return static_cast<Player>(player) == Player::Black;
}

} // namespace tengen::gui
