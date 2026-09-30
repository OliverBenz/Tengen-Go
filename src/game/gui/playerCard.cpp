#include "playerCard.hpp"

#include "gui/resources.hpp"

#include <QApplication>
#include <QGraphicsDropShadowEffect>
#include <QHBoxLayout>
#include <QLabel>
#include <QPixmap>

namespace tengen::gui {

static constexpr int STONE_SIZE = 28; //!< Stone image edge [px].

PlayerCard::PlayerCard(const bool mirrored, QWidget* parent)
    : QFrame(parent) {
	setObjectName("playerCard"); // Keeps the style sheet on the frame, off its labels.
	setMinimumWidth(140);

	// Setup stone image
	m_stone = new QLabel(this);
	m_stone->setFixedSize(STONE_SIZE, STONE_SIZE);

	// Setup player name
	m_name         = new QLabel(this);
	QFont nameFont = m_name->font();
	nameFont.setBold(true);
	m_name->setFont(nameFont);

	// Setup layout
	auto* layout = new QHBoxLayout(this);
	layout->setContentsMargins(10, 6, 10, 6);
	layout->setSpacing(8);
	if (mirrored) {
		m_name->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
		layout->addWidget(m_name, 1);
		layout->addWidget(m_stone);
	} else {
		layout->addWidget(m_stone);
		layout->addWidget(m_name, 1);
	}

	// Add shadow
	auto* shadow = new QGraphicsDropShadowEffect(this);
	shadow->setBlurRadius(12);
	shadow->setOffset(0, 2);
	shadow->setColor(QColor(0, 0, 0, 60));
	setGraphicsEffect(shadow);

	setPlayer(Player::Black, {});
	updateStyle();
}

void PlayerCard::setPlayer(const Player colour, const QString& name) {
	m_colour = colour;
	m_name->setText(name);

	const qreal ratio = devicePixelRatioF();
	QPixmap stone     = QPixmap::fromImage(loadStone(colour));
	if (!stone.isNull()) {
		stone = stone.scaled(QSize(STONE_SIZE, STONE_SIZE) * ratio, Qt::KeepAspectRatio, Qt::SmoothTransformation);
		stone.setDevicePixelRatio(ratio);
	}
	m_stone->setPixmap(stone);
}

Player PlayerCard::colour() const {
	return m_colour;
}

void PlayerCard::setActive(const bool active) {
	if (active == m_active) {
		return;
	}
	m_active = active;
	updateStyle();
}

void PlayerCard::updateStyle() {
	static constexpr float TINT = 0.18f; //!< How much of the highlight colour the side to move gets.

	// Our own style sheet writes its background into palette(), so read the application's untouched colours instead.
	// A light wash of the highlight colour, so the side to move stands out without shouting.
	const QPalette colours = QApplication::palette(this);
	QColor background      = colours.color(QPalette::Base);
	if (m_active) {
		const QColor highlight = colours.color(QPalette::Highlight);
		background.setRgbF(background.redF() + (highlight.redF() - background.redF()) * TINT,
		                   background.greenF() + (highlight.greenF() - background.greenF()) * TINT,
		                   background.blueF() + (highlight.blueF() - background.blueF()) * TINT);
	}

	setStyleSheet(QString("QFrame#playerCard { border: 1px solid palette(mid); border-radius: 6px; background: %1; }")
	                      .arg(background.name()));
}

} // namespace tengen::gui
