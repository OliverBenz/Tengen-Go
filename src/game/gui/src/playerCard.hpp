#pragma once

#include "model/player.hpp"

#include <QFrame>
#include <QString>

class QLabel;

namespace tengen::gui {

//! Small framed box naming one player next to a small stone image. Lights up while that side is to move.
class PlayerCard : public QFrame {
	Q_OBJECT

public:
	//! A 'mirrored' card puts its stone on the right side of the box so two cards framing the status face each other.
	PlayerCard(bool mirrored, QWidget* parent = nullptr);

	void setPlayer(Player colour, const QString& name);
	Player colour() const;

	void setActive(bool active); //!< Tint the background while this side is to move.

private:
	void updateStyle();

private:
	QLabel* m_stone{nullptr};
	QLabel* m_name{nullptr};
	Player m_colour{Player::Black};
	bool m_active{false};
};

} // namespace tengen::gui
