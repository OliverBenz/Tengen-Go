#pragma once

#include "gui/resources.hpp"

#include <QSoundEffect>

namespace tengen::gui {

//! Plays the application's short sound effects. Keeps one QSoundEffect per sound alive so replaying it doesn't reload the file.
class SoundPlayer {
public:
	SoundPlayer();

	stoneStyle::Sound stoneSound() const;
	void setStoneSound(stoneStyle::Sound sound); //!< Sound played on stone placement.

	void playStonePlace();

private:
	stoneStyle::Sound m_stoneSound{stoneStyle::defaultSound()};
	QSoundEffect m_stonePlace;
};

} // namespace tengen::gui
