#include "gui/soundPlayer.hpp"

#include <QUrl>

namespace tengen::gui {

SoundPlayer::SoundPlayer() {
	m_stonePlace.setSource(QUrl::fromLocalFile(soundFile(m_stoneSound)));
}

StoneSound SoundPlayer::stoneSound() const {
	return m_stoneSound;
}

void SoundPlayer::setStoneSound(const StoneSound sound) {
	m_stoneSound = sound;
	m_stonePlace.setSource(QUrl::fromLocalFile(soundFile(sound)));
}

void SoundPlayer::playStonePlace() {
	m_stonePlace.play();
}

} // namespace tengen::gui
