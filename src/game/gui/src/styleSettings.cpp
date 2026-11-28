#include "gui/styleSettings.hpp"

#include <QSettings>
#include <algorithm>

namespace tengen::gui {
namespace {

constexpr char BOARD_TEXTURE[]    = "style/boardTexture";
constexpr char SHOW_COORDINATES[] = "style/showCoordinates";
constexpr char STONE_TEXTURE[]    = "style/stoneTexture";
constexpr char STONE_SOUND[]      = "style/stoneSound";

//! The choice whose settings key is stored under name. Fallback if missing or unknown.
template <typename Choice>
Choice readChoice(const QSettings& settings, const QString& name, const QList<Choice>& choices, const Choice fallback) {
	const QString stored = settings.value(name).toString();
	const auto found     = std::ranges::find(choices, stored, [](const Choice choice) { return settingsKey(choice); });
	return found != choices.end() ? *found : fallback;
}

//! Only accepts the values QSettings writes, so a typo doesn't count as true.
bool readBool(const QSettings& settings, const QString& name, const bool fallback) {
	const QString stored = settings.value(name).toString();
	if (stored == QLatin1String("true")) {
		return true;
	}
	if (stored == QLatin1String("false")) {
		return false;
	}
	return fallback;
}

} // namespace

Style loadStyle(const QSettings& settings) {
	const Style fallback{};
	return Style{
	        .boardTexture    = readChoice(settings, BOARD_TEXTURE, boardTextures(), fallback.boardTexture),
	        .showCoordinates = readBool(settings, SHOW_COORDINATES, fallback.showCoordinates),
	        .stoneTexture    = readChoice(settings, STONE_TEXTURE, stoneTextures(), fallback.stoneTexture),
	        .stoneSound      = readChoice(settings, STONE_SOUND, stoneSounds(), fallback.stoneSound),
	};
}

void saveStyle(QSettings& settings, const Style& style) {
	settings.setValue(BOARD_TEXTURE, settingsKey(style.boardTexture));
	settings.setValue(SHOW_COORDINATES, style.showCoordinates);
	settings.setValue(STONE_TEXTURE, settingsKey(style.stoneTexture));
	settings.setValue(STONE_SOUND, settingsKey(style.stoneSound));
}

} // namespace tengen::gui
