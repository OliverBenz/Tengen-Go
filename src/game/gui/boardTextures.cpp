#include "gui/boardTextures.hpp"

#include <QDir>
#include <QImageReader>

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

} // namespace tengen::gui
