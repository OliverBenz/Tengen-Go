#pragma once

#include "gui/resources.hpp"

#include <QDialog>
#include <QFutureWatcher>
#include <QImage>

class QCheckBox;
class QComboBox;
class QListWidget;

namespace tengen::gui {

class BoardWidget;

//! Lets the user pick one of the board textures while previewing the board next to them.
class BoardStyleDialog : public QDialog {
	Q_OBJECT

public:
	explicit BoardStyleDialog(const Style& currentStyle, QWidget* parent = nullptr);
	~BoardStyleDialog() override;

	Style currentStyle() const; //!< Style selected in the dialog.
	BoardWidget& boardWidget(); //!< The board preview. Shows whatever board it is given.

protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

private:
	void addTexturePreviews();
	void selectTexture(BoardTexture texture); //!< Falls back to plain if the texture is not listed.
	void fitPreviewCells();                   //!< Stretch the preview cells over the whole list width.

private:
	QListWidget* m_textures{nullptr};       //!< List of texture preview and names.
	QCheckBox* m_coordinates{nullptr};      //!< Selection for whether to show coordinates or not.
	QComboBox* m_sounds{nullptr};           //!< Selection of the stone placement sound.
	BoardWidget* m_board{nullptr};          //!< The board preview.
	QFutureWatcher<QImage> m_previewLoader; //!< Loads the texture previews in the background.
	Style m_previewStyle;                   //!< Holds the style being edited; only reaches the app once accepted.
};

} // namespace tengen::gui
