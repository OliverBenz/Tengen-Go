#include "gui/resources.hpp"

#include <QCoreApplication>
#include <QImageReader>
#include <algorithm>
#include <array>
#include <cassert>

namespace tengen::gui {
namespace {

//! The build copies the resource folder to resource/ next to the executable.
QString resourcePath(const QString& relative) {
	return QCoreApplication::applicationDirPath() + QStringLiteral("/resource/") + relative;
}

QImage readImage(const QString& relative) {
	QImageReader reader(resourcePath(relative));
	reader.setAutoTransform(true);
	return reader.read();
}

//! Qt's smooth scaling and drawing only work directly on these formats. Anything else gets converted on every call.
QImage toFastFormat(const QImage& image) {
	return image.convertToFormat(image.hasAlphaChannel() ? QImage::Format_ARGB32_Premultiplied : QImage::Format_RGB32);
}

//! A board texture and the file behind it. The only place that knows texture files.
struct TextureFile {
	BoardTexture texture;
	const char* name; //!< Display name, translated on use.
	const char* file; //!< File in the boards folder. Null for the plain board.
};

// Display order of the style settings.
constexpr std::array TEXTURES{
        TextureFile{BoardTexture::Plain, QT_TRANSLATE_NOOP("Style", "Plain"), nullptr},
        TextureFile{BoardTexture::HoneyWood, QT_TRANSLATE_NOOP("Style", "Honey Wood"), "Wood095_2K-PNG_Color.png"},
        TextureFile{BoardTexture::LightWood, QT_TRANSLATE_NOOP("Style", "Light Wood"), "Wood094_2K-PNG_Color.png"},
        TextureFile{BoardTexture::Afromosia, QT_TRANSLATE_NOOP("Style", "Afromosia"), "2K_afromosia_basecolor.png"},
        TextureFile{BoardTexture::GreyWood, QT_TRANSLATE_NOOP("Style", "Grey Wood"), "1K-wood_fine_8-diffuse.jpg"},
        TextureFile{BoardTexture::Anime, QT_TRANSLATE_NOOP("Style", "Anime"), "anime_board.svg"},
};

const TextureFile& textureFile(const BoardTexture texture) {
	const auto entry = std::ranges::find(TEXTURES, texture, &TextureFile::texture);
	assert(entry != TEXTURES.end() && "Every BoardTexture needs an entry in TEXTURES.");
	return *entry;
}

//! A stone set and the files behind it, one per player colour. The only place that knows stone files.
struct StoneTextureFile {
	StoneTexture texture;
	const char* name;  //!< Display name, translated on use.
	const char* black; //!< File in the stones folder for the black stone.
	const char* white; //!< File in the stones folder for the white stone.
};

// Display order of the style settings.
constexpr std::array STONE_TEXTURES{
        StoneTextureFile{StoneTexture::Anime, QT_TRANSLATE_NOOP("Style", "Anime"), "stones/anime_black.png", "stones/anime_white.png"},
};

const StoneTextureFile& stoneTextureFile(const StoneTexture texture) {
	const auto entry = std::ranges::find(STONE_TEXTURES, texture, &StoneTextureFile::texture);
	assert(entry != STONE_TEXTURES.end() && "Every StoneTexture needs an entry in STONE_TEXTURES.");
	return *entry;
}

//! A stone placement sound and the file behind it.
struct SoundFile {
	StoneSound sound;
	const char* name; //!< Display name, translated on use.
	const char* file; //!< File in the audio folder.
};

//! Display order of the style settings.
constexpr std::array SOUNDS{
        SoundFile{StoneSound::One, QT_TRANSLATE_NOOP("Style", "Place 1"), "stone_1.wav"},
        SoundFile{StoneSound::Two, QT_TRANSLATE_NOOP("Style", "Place 2"), "stone_2.wav"},
};

const SoundFile& soundEntry(const StoneSound sound) {
	const auto entry = std::ranges::find(SOUNDS, sound, &SoundFile::sound);
	assert(entry != SOUNDS.end() && "Every StoneSound needs an entry in SOUNDS.");
	return *entry;
}

} // namespace

QList<BoardTexture> boardTextures() {
	QList<BoardTexture> result;
	for (const auto& entry: TEXTURES) {
		result.append(entry.texture);
	}
	return result;
}

BoardTexture defaultBoardTexture() {
	return BoardTexture::HoneyWood;
}

QString displayName(const BoardTexture texture) {
	return QCoreApplication::translate("Style", textureFile(texture).name);
}

QImage loadBoardTexture(const BoardTexture texture) {
	const char* file = textureFile(texture).file;
	if (!file) {
		return {};
	}
	const QImage image = readImage(QStringLiteral("boards/") + QString::fromLatin1(file));

	// The board is square: use the center of the image rather than stretching it.
	const int side = std::min(image.width(), image.height());
	return toFastFormat(image.copy((image.width() - side) / 2, (image.height() - side) / 2, side, side));
}

QList<StoneTexture> stoneTextures() {
	QList<StoneTexture> result;
	for (const auto& entry: STONE_TEXTURES) {
		result.append(entry.texture);
	}
	return result;
}

StoneTexture defaultStoneTexture() {
	return StoneTexture::Anime;
}

QString displayName(const StoneTexture texture) {
	return QCoreApplication::translate("Style", stoneTextureFile(texture).name);
}

QImage loadStoneTexture(const StoneTexture texture, const Player player) {
	const StoneTextureFile& entry = stoneTextureFile(texture);
	return toFastFormat(readImage(QString::fromLatin1(player == Player::Black ? entry.black : entry.white)));
}

QList<StoneSound> stoneSounds() {
	QList<StoneSound> result;
	for (const auto& entry: SOUNDS) {
		result.append(entry.sound);
	}
	return result;
}

StoneSound defaultStoneSound() {
	return StoneSound::One;
}

QString soundFile(const StoneSound sound) {
	return resourcePath(QStringLiteral("audio/") + QString::fromLatin1(soundEntry(sound).file));
}

QString displayName(const StoneSound sound) {
	return QCoreApplication::translate("Style", soundEntry(sound).name);
}


namespace gameRules {

QString displayName(const RuleSet ruleSet) {
	switch (ruleSet) {
	case RuleSet::Japanese:
		return QCoreApplication::translate("gameRules", "Japanese");
	case RuleSet::Chinese:
		return QCoreApplication::translate("gameRules", "Chinese");
	case RuleSet::Korean:
		return QCoreApplication::translate("gameRules", "Korean");
	}

	assert(false);
	return {};
}

QString displayName(const Scoring scoring) {
	switch (scoring) {
	case Scoring::Territory:
		return QCoreApplication::translate("gameRules", "Territory");
	case Scoring::Area:
		return QCoreApplication::translate("gameRules", "Area");
	}

	assert(false);
	return {};
}

QString displayName(const Ko ko) {
	switch (ko) {
	case Ko::Simple:
		return QCoreApplication::translate("gameRules", "Simple");
	case Ko::Situational:
		return QCoreApplication::translate("gameRules", "Situational");
	case Ko::Positional:
		return QCoreApplication::translate("gameRules", "Positional");
	}

	assert(false);
	return {};
}

} // namespace gameRules

} // namespace tengen::gui
