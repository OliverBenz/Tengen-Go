#include "gui/boardTextures.hpp"

#include <QDir>
#include <QImageReader>
#include <algorithm>

namespace tengen::gui {

QStringList boardTexturePaths() {
	QStringList nameFilters;
	for (const auto& format: QImageReader::supportedImageFormats()) {
		nameFilters.append("*." + QString::fromLatin1(format));
	}

	const QDir dir(GUI_RESOURCES_DIR "/board_textures");
	QStringList paths;
	for (const auto& file: dir.entryInfoList(nameFilters, QDir::Files, QDir::Name)) {
		paths.append(file.absoluteFilePath());
	}
	return paths;
}

QImage loadBoardTexture(const QString& path) {
	QImageReader reader(path);
	reader.setAutoTransform(true);
	const QImage image = reader.read();

	// The board is square: use the center of the image rather than stretching it.
	const int side = std::min(image.width(), image.height());
	return image.copy((image.width() - side) / 2, (image.height() - side) / 2, side, side);
}

} // namespace tengen::gui
