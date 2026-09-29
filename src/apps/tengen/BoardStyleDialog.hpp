#pragma once

#include <QDialog>
#include <QString>

class QListWidget;

namespace tengen::gui {

class BoardWidget;

//! Lets the user pick one of the board textures while previewing the board next to them.
class BoardStyleDialog : public QDialog {
	Q_OBJECT

public:
	explicit BoardStyleDialog(const QString& currentTexture, QWidget* parent = nullptr);

	QString texturePath() const; //!< Selected texture. Empty for the plain board colour.

private:
	void addTexturePreviews();
	void selectTexture(const QString& path);

private:
	QListWidget* m_textures = nullptr;
	BoardWidget* m_board    = nullptr;
};

} // namespace tengen::gui
