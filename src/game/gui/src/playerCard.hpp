#pragma once

#include "gui/resources.hpp"
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
	PlayerCard(const Style& initialStyle, bool mirrored, QWidget* parent = nullptr);

	void setPlayer(Player colour, const QString& name);
	Player colour() const;

	void setActive(bool active); //!< Tint the background while this side is to move.

public slots:
	void applyStyle(const Style& style); //!< Pick up the stone texture and redraw the icon.

private:
	void updateStyle();     //!< Re-tint the card's background.
	void updateStoneIcon(); //!< Redraw the stone image for the current colour and texture.

private:
	QLabel* m_stone{nullptr};
	QLabel* m_name{nullptr};
	Player m_colour{Player::Black};
	StoneTexture m_stoneTexture;
	bool m_active{false};
};

} // namespace tengen::gui
