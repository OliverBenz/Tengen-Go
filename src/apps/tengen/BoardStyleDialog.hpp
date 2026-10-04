#pragma once

#include "gui/resources.hpp"

#include <QDialog>
#include <QFutureWatcher>
#include <QImage>

class QCheckBox;
class QListWidget;

namespace tengen::gui {

class BoardWidget;

//! Lets the user pick one of the board textures while previewing the board next to them.
class BoardStyleDialog : public QDialog {
	Q_OBJECT

public:
	BoardStyleDialog(boardStyle::Texture currentTexture, bool showCoordinates, QWidget* parent = nullptr);
	~BoardStyleDialog() override;

	boardStyle::Texture texture() const; //!< Selected texture.
	bool showCoordinates() const;        //!< Whether the board labels its lines.

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

private:
	void addTexturePreviews();
	void selectTexture(boardStyle::Texture texture); //!< Falls back to plain if the texture is not listed.
	void fitPreviewCells();                          //!< Stretch the preview cells over the whole list width.

private:
	QListWidget* m_textures{nullptr};       //!< List of texture preview and names.
	QCheckBox* m_coordinates{nullptr};      //!< Selection for whether to show coordinates or not.
	BoardWidget* m_board{nullptr};          //!< The board preview.
	QFutureWatcher<QImage> m_previewLoader; //!< Loads the texture previews in the background.
};

} // namespace tengen::gui
