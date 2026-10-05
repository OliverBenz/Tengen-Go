#include "gui/resources.hpp"

#include <QCoreApplication>
#include <QImageReader>
#include <algorithm>
#include <array>
#include <cassert>

namespace tengen::gui {
namespace {

QString resourcePath(const QString& relative) {
	return QStringLiteral(GUI_RESOURCES_DIR "/") + relative;
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

} // namespace

namespace boardStyle {
namespace {

//! A board texture and the file behind it. The only place that knows texture files.
struct TextureFile {
	Texture texture;
	const char* name; //!< Display name, translated on use.
	const char* file; //!< File in the board_textures folder. Null for the plain board.
};

// Display order of the style settings.
constexpr std::array TEXTURES{
        TextureFile{Texture::Plain, QT_TRANSLATE_NOOP("boardStyle", "Plain"), nullptr},
        TextureFile{Texture::HoneyWood, QT_TRANSLATE_NOOP("boardStyle", "Honey Wood"), "Wood095_2K-PNG_Color.png"},
        TextureFile{Texture::LightWood, QT_TRANSLATE_NOOP("boardStyle", "Light Wood"), "Wood094_2K-PNG_Color.png"},
        TextureFile{Texture::Afromosia, QT_TRANSLATE_NOOP("boardStyle", "Afromosia"), "2K_afromosia_basecolor.png"},
        TextureFile{Texture::GreyWood, QT_TRANSLATE_NOOP("boardStyle", "Grey Wood"), "1K-wood_fine_8-diffuse.jpg"},
        TextureFile{Texture::Anime, QT_TRANSLATE_NOOP("boardStyle", "Anime"), "anime_board.svg"},
};

const TextureFile& textureFile(const Texture texture) {
	const auto entry = std::ranges::find(TEXTURES, texture, &TextureFile::texture);
	assert(entry != TEXTURES.end() && "Every Texture needs an entry in TEXTURES.");
	return *entry;
}

} // namespace

QList<Texture> textures() {
	QList<Texture> result;
	for (const auto& entry: TEXTURES) {
		result.append(entry.texture);
	}
	return result;
}

Texture defaultTexture() {
	return Texture::HoneyWood;
}

QString displayName(const Texture texture) {
	return QCoreApplication::translate("boardStyle", textureFile(texture).name);
}

QImage loadTexture(const Texture texture) {
	const char* file = textureFile(texture).file;
	if (!file) {
		return {};
	}
	const QImage image = readImage(QStringLiteral("board_textures/") + QString::fromLatin1(file));

	// The board is square: use the center of the image rather than stretching it.
	const int side = std::min(image.width(), image.height());
	return toFastFormat(image.copy((image.width() - side) / 2, (image.height() - side) / 2, side, side));
}

} // namespace boardStyle

namespace stoneStyle {

QImage loadTexture(const Player player) {
	return toFastFormat(readImage(player == Player::Black ? QStringLiteral("anime_black.png") : QStringLiteral("anime_white.png")));
}

} // namespace stoneStyle

namespace sound {

QString stonePlace() {
	return resourcePath(QStringLiteral("audio/stone_1.wav"));
}

} // namespace sound

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
