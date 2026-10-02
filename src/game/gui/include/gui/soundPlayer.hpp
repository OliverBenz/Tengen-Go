#pragma once

#include <QSoundEffect>

namespace tengen::gui {

//! Plays the application's short sound effects. Keeps one QSoundEffect per sound alive so replaying it doesn't reload the file.
class SoundPlayer {
public:
	SoundPlayer();

	void playStonePlace();

private:
	QSoundEffect m_stonePlace;
};

} // namespace tengen::gui
