#pragma once

#include <QDialog>
#include <QFutureWatcher>
#include <QImage>
#include <QString>

class QListWidget;

namespace tengen::gui {

class BoardWidget;

//! Lets the user pick one of the board textures while previewing the board next to them.
class BoardStyleDialog : public QDialog {
	Q_OBJECT

public:
	explicit BoardStyleDialog(const QString& currentTexture, QWidget* parent = nullptr);
	~BoardStyleDialog() override;

	QString texturePath() const; //!< Selected texture. Empty for the plain board colour.

private:
	void addTexturePreviews();
	void selectTexture(const QString& path);

private:
	QListWidget* m_textures = nullptr;
	BoardWidget* m_board    = nullptr;
	QFutureWatcher<QImage> m_previewLoader; //!< Loads the texture previews in the background.
};

} // namespace tengen::gui
