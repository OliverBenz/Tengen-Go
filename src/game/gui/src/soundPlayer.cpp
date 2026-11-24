#include "gui/soundPlayer.hpp"

#include <QUrl>

namespace tengen::gui {

SoundPlayer::SoundPlayer() {
	m_stonePlace.setSource(QUrl::fromLocalFile(stoneStyle::soundFile(m_stoneSound)));
}

stoneStyle::Sound SoundPlayer::stoneSound() const {
	return m_stoneSound;
}

void SoundPlayer::setStoneSound(const stoneStyle::Sound sound) {
	m_stoneSound = sound;
	m_stonePlace.setSource(QUrl::fromLocalFile(stoneStyle::soundFile(sound)));
}

void SoundPlayer::playStonePlace() {
	m_stonePlace.play();
}

} // namespace tengen::gui
