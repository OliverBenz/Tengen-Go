#pragma once

#include "model/gameRules.hpp"
#include "model/player.hpp"

#include <QColor>
#include <QImage>
#include <QList>
#include <QString>
#include <cstdint>

// Contains the resources for board styles, stone styles, sounds and the names of the game rules.
namespace tengen::gui {

inline constexpr QColor PLAIN_BOARD_COLOUR{220, 179, 92}; //!< Board colour of the plain texture.

//! Board backgrounds the application ships. Which file backs a texture stays inside the resources.
enum class BoardTexture : std::uint8_t {
	Plain, //!< No image: the board is filled with PLAIN_BOARD_COLOUR.
	HoneyWood,
	LightWood,
	Afromosia,
	GreyWood,
	Anime,
};

QList<BoardTexture> boardTextures();           //!< All board textures in display order, Plain first.
BoardTexture defaultBoardTexture();            //!< Texture a new board starts with.
QString displayName(BoardTexture texture);     //!< Name to show the user.
QImage loadBoardTexture(BoardTexture texture); //!< Centre square of the texture, as the board shows it. Null for Plain or if unreadable.


//! Stone sets the application ships. Which files back a texture stays inside the resources.
enum class StoneTexture : std::uint8_t {
	Anime,
};

QList<StoneTexture> stoneTextures();                          //!< All stone textures in display order.
StoneTexture defaultStoneTexture();                           //!< Stones a new board starts with.
QString displayName(StoneTexture texture);                    //!< Name to show the user.
QImage loadStoneTexture(StoneTexture texture, Player player); //!< Stone image of a player at full resolution. Callers scale it to their size. Null if unreadable.


//! Stone placement sounds the application ships.
enum class StoneSound {
	One,
	Two,
};

QList<StoneSound> stoneSounds();        //!< All stone sounds in display order.
StoneSound defaultStoneSound();         //!< Sound a new board starts with.
QString soundFile(StoneSound sound);    //!< Path to the given stone placement sound effect.
QString displayName(StoneSound sound);  //!< Name to show the user.


namespace gameRules {

QString displayName(RuleSet ruleSet); //!< Returns the user-facing string for the GUI.
QString displayName(Scoring scoring); //!< Returns the user-facing string for the GUI.
QString displayName(Ko ko);           //!< Returns the user-facing string for the GUI.

} // namespace gameRules

} // namespace tengen::gui
