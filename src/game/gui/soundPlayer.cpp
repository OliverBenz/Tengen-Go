#include "gui/soundPlayer.hpp"
#include "gui/resources.hpp"

#include <QUrl>

namespace tengen::gui {

SoundPlayer::SoundPlayer() {
	m_stonePlace.setSource(QUrl::fromLocalFile(sound::stonePlace()));
}

void SoundPlayer::playStonePlace() {
	m_stonePlace.play();
}

} // namespace tengen::gui
