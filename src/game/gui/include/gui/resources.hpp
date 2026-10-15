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


//! Stone placement sounds the application ships.
enum class Sound {
	One,
	Two,
};

QList<Sound> sounds();            //!< All stone sounds in display order.
Sound defaultSound();             //!< Sound a new board starts with.
QString soundFile(Sound sound);   //!< Path to the given stone placement sound effect.
QString displayName(Sound sound); //!< Name to show the user.

} // namespace stoneStyle


namespace gameRules {

QString displayName(RuleSet ruleSet); //!< Returns the user-facing string for the GUI.
QString displayName(Scoring scoring); //!< Returns the user-facing string for the GUI.
QString displayName(Ko ko);           //!< Returns the user-facing string for the GUI.

} // namespace gameRules

} // namespace tengen::gui
