#pragma once

#include "gui/resources.hpp"

#include <QSoundEffect>

namespace tengen::gui {

//! Plays the application's short sound effects. Keeps one QSoundEffect per sound alive so replaying it doesn't reload the file.
class SoundPlayer {
public:
	SoundPlayer();

	StoneSound stoneSound() const;
	void setStoneSound(StoneSound sound); //!< Sound played on stone placement.

	void playStonePlace();

private:
	StoneSound m_stoneSound{defaultStoneSound()};
	QSoundEffect m_stonePlace;
};

} // namespace tengen::gui
