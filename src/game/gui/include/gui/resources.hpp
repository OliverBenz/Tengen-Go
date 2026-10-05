#pragma once

#include "model/gameRules.hpp"
#include "model/player.hpp"

#include <QColor>
#include <QImage>
#include <QList>
#include <QString>

// Contains the resources for board styles, stone styles, sounds and the names of the game rules.
namespace tengen::gui {
namespace boardStyle {

inline constexpr QColor PLAIN_COLOUR{220, 179, 92}; //!< Board colour of the plain texture.

//! Board backgrounds the application ships. Which file backs a texture stays inside the resources.
enum class Texture {
	Plain, //!< No image: the board is filled with PLAIN_COLOUR.
	HoneyWood,
	LightWood,
	Afromosia,
	GreyWood,
	Anime,
};

QList<Texture> textures();            //!< All board textures in display order, Plain first.
Texture defaultTexture();             //!< Texture a new board starts with.
QString displayName(Texture texture); //!< Name to show the user.
QImage loadTexture(Texture texture);  //!< Centre square of the texture, as the board shows it. Null for Plain or if unreadable.

} // namespace boardStyle


namespace stoneStyle {

//! Stone image of a player at full resolution. Callers scale it to their size. Null if unreadable.
QImage loadTexture(Player player);

} // namespace stoneStyle


namespace sound {

//! Path to the stone placement sound effect, for gui::SoundPlayer to load.
QString stonePlace();

} // namespace sound


namespace gameRules {

QString displayName(RuleSet ruleSet); //!< Returns the user-facing string for the GUI.
QString displayName(Scoring scoring); //!< Returns the user-facing string for the GUI.
QString displayName(Ko ko);           //!< Returns the user-facing string for the GUI.

} // namespace gameRules

} // namespace tengen::gui
