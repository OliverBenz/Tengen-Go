#include "BoardStyleDialog.hpp"

#include "gui/boardTextures.hpp"
#include "gui/boardWidget.hpp"

#include <QDialogButtonBox>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QImageReader>
#include <QListWidget>
#include <QVBoxLayout>

namespace tengen::gui {

static constexpr int PREVIEW_SIZE = 128; //!< Edge length of a texture thumbnail [px].

//! Qt tints selected icons with the highlight colour, which would falsify the texture colours.
static QIcon untintedIcon(const QPixmap& pixmap) {
	QIcon icon(pixmap);
	icon.addPixmap(pixmap, QIcon::Selected);
	return icon;
}

BoardStyleDialog::BoardStyleDialog(const QString& currentTexture, QWidget* parent)
    : QDialog(parent) {
	setWindowTitle(tr("Board Style"));
	resize(900, 620);

	m_board = new BoardWidget(this);
	m_board->setBoard(Board(19u));

	m_textures = new QListWidget(this);
	m_textures->setViewMode(QListView::IconMode);
	m_textures->setIconSize({PREVIEW_SIZE, PREVIEW_SIZE});
	m_textures->setFlow(QListView::TopToBottom);
	m_textures->setWrapping(false);
	m_textures->setMovement(QListView::Static);
	m_textures->setWordWrap(true);
	m_textures->setFixedWidth(PREVIEW_SIZE + 60);
	connect(m_textures, &QListWidget::currentItemChanged, this, [this] { m_board->setBackgroundTexture(texturePath()); });
	addTexturePreviews();
	selectTexture(currentTexture);

	auto* content = new QHBoxLayout();
	content->addWidget(m_textures);
	content->addWidget(m_board, 1);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addLayout(content, 1);
	mainLayout->addWidget(buttons);
}

QString BoardStyleDialog::texturePath() const {
	const auto* item = m_textures->currentItem();
	return item ? item->data(Qt::UserRole).toString() : QString{};
}

void BoardStyleDialog::addTexturePreviews() {
	QPixmap plain(PREVIEW_SIZE, PREVIEW_SIZE);
	plain.fill(PLAIN_BOARD_COLOUR);
	new QListWidgetItem(untintedIcon(plain), tr("Plain"), m_textures); // Empty path: no texture.

	for (const auto& path: boardTexturePaths()) {
		QImageReader reader(path);
		reader.setAutoTransform(true);
		const QImage preview = reader.read().scaled(PREVIEW_SIZE, PREVIEW_SIZE, Qt::KeepAspectRatio, Qt::SmoothTransformation);
		if (preview.isNull()) {
			continue; // Unreadable file: nothing to preview.
		}

		auto* item = new QListWidgetItem(untintedIcon(QPixmap::fromImage(preview)), QFileInfo(path).completeBaseName(), m_textures);
		item->setData(Qt::UserRole, path);
	}
}

void BoardStyleDialog::selectTexture(const QString& path) {
	for (int row = 0; row != m_textures->count(); ++row) {
		if (m_textures->item(row)->data(Qt::UserRole).toString() == path) {
			m_textures->setCurrentRow(row);
			return;
		}
	}
	m_textures->setCurrentRow(0); // Texture no longer exists: fall back to plain.
}

} // namespace tengen::gui
