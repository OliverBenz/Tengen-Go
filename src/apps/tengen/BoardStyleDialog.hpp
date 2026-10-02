#pragma once

#include "gui/resources.hpp"

#include <QDialog>
#include <QFutureWatcher>
#include <QImage>

class QListWidget;

namespace tengen::gui {

class BoardWidget;

//! Lets the user pick one of the board textures while previewing the board next to them.
class BoardStyleDialog : public QDialog {
	Q_OBJECT

public:
	explicit BoardStyleDialog(boardStyle::Texture currentTexture, QWidget* parent = nullptr);
	~BoardStyleDialog() override;

	boardStyle::Texture texture() const; //!< Selected texture.

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

private:
	void addTexturePreviews();
	void selectTexture(boardStyle::Texture texture); //!< Falls back to plain if the texture is not listed.
	void fitPreviewCells();                          //!< Stretch the preview cells over the whole list width.

private:
	QListWidget* m_textures = nullptr;
	BoardWidget* m_board    = nullptr;
	QFutureWatcher<QImage> m_previewLoader; //!< Loads the texture previews in the background.
};

} // namespace tengen::gui
