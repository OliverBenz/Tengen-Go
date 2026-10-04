#include "BoardStyleDialog.hpp"

#include "gui/boardWidget.hpp"

#include <QCheckBox>
#include <QDialogButtonBox>
#include <QEvent>
#include <QHBoxLayout>
#include <QListWidget>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentMap>

namespace tengen::gui {

static constexpr int PREVIEW_SIZE = 128; //!< Edge length of a texture thumbnail [px].
static constexpr int LIST_PADDING = 4;   //!< Space above and below each preview inside its cell [px] (first+last element should not touch list border).

//! Qt tints selected icons with the highlight colour, which would falsify the texture colours.
static QIcon untintedIcon(const QPixmap& pixmap) {
	QIcon icon(pixmap);
	icon.addPixmap(pixmap, QIcon::Selected);
	return icon;
}

static boardStyle::Texture textureOf(const QListWidgetItem& item) {
	return static_cast<boardStyle::Texture>(item.data(Qt::UserRole).toInt());
}

BoardStyleDialog::BoardStyleDialog(const boardStyle::Texture currentTexture, const bool showCoordinates, QWidget* parent)
    : QDialog(parent) {
	setWindowTitle(tr("Board Style"));
	resize(900, 620);

	m_board = new BoardWidget(this);
	m_board->setBoard(Board(19u));
	m_board->setShowCoordinates(showCoordinates);

	m_textures = new QListWidget(this);
	m_textures->setViewMode(QListView::IconMode);
	m_textures->setIconSize({PREVIEW_SIZE, PREVIEW_SIZE});
	m_textures->setGridSize({PREVIEW_SIZE + 40, PREVIEW_SIZE + fontMetrics().height() + 10 + 2 * LIST_PADDING});
	m_textures->setFlow(QListView::TopToBottom);
	m_textures->setWrapping(false);
	m_textures->setMovement(QListView::Static);
	m_textures->setWordWrap(true);
	m_textures->setFixedWidth(PREVIEW_SIZE + 60);
	m_textures->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
	m_textures->viewport()->installEventFilter(this); // The width shrinks while the scroll bar shows.
	connect(m_textures, &QListWidget::currentItemChanged, this, [this] { m_board->setBackgroundTexture(texture()); });
	addTexturePreviews();
	selectTexture(currentTexture);

	auto* content = new QHBoxLayout();
	content->addWidget(m_textures);
	content->addWidget(m_board, 1);

	m_coordinates = new QCheckBox(tr("Show coordinates"), this);
	m_coordinates->setChecked(showCoordinates);
	connect(m_coordinates, &QCheckBox::toggled, m_board, &BoardWidget::setShowCoordinates);

	auto* buttons = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel, this);
	connect(buttons, &QDialogButtonBox::accepted, this, &QDialog::accept);
	connect(buttons, &QDialogButtonBox::rejected, this, &QDialog::reject);

	auto* mainLayout = new QVBoxLayout(this);
	mainLayout->addLayout(content, 1);
	mainLayout->addWidget(m_coordinates);
	mainLayout->addWidget(buttons);
}

BoardStyleDialog::~BoardStyleDialog() {
	m_previewLoader.cancel(); // Closed before all previews loaded: skip the rest.
}

boardStyle::Texture BoardStyleDialog::texture() const {
	const auto* item = m_textures->currentItem();
	return item ? textureOf(*item) : boardStyle::Texture::Plain;
}

bool BoardStyleDialog::showCoordinates() const {
	return m_coordinates->isChecked();
}

bool BoardStyleDialog::eventFilter(QObject* watched, QEvent* event) {
	if (watched == m_textures->viewport() && event->type() == QEvent::Resize) {
		fitPreviewCells();
	}
	return QDialog::eventFilter(watched, event);
}

//! Flowing top to bottom, Qt centres an item in its cell vertically only and shrinks it to its content.
//! Stretching cells and items to the full list width lets the delegate centre icon and name.
//! Items are shorter than their cell by the padding, which Qt splits above and below the item.
void BoardStyleDialog::fitPreviewCells() {
	const QSize cell{m_textures->viewport()->width(), m_textures->gridSize().height()};
	const QSize item{cell.width(), cell.height() - 2 * LIST_PADDING};
	m_textures->setGridSize(cell);
	for (int row = 0; row != m_textures->count(); ++row) {
		m_textures->item(row)->setSizeHint(item);
	}
}

void BoardStyleDialog::addTexturePreviews() {
	const qreal ratio = devicePixelRatioF();
	const int side    = qRound(PREVIEW_SIZE * ratio); // Physical pixels, so previews stay sharp on scaled displays.

	const auto filled = [side, ratio](const QColor& colour) {
		QPixmap pixmap(side, side);
		pixmap.fill(colour);
		pixmap.setDevicePixelRatio(ratio);
		return untintedIcon(pixmap);
	};

	// The plain board shows its colour right away. Images stay blank until loaded, which keeps the item size stable.
	QList<boardStyle::Texture> images;
	QList<QListWidgetItem*> items;
	for (const boardStyle::Texture texture: boardStyle::textures()) {
		const bool plain = texture == boardStyle::Texture::Plain;
		auto* item       = new QListWidgetItem(filled(plain ? boardStyle::PLAIN_COLOUR : Qt::transparent), boardStyle::displayName(texture), m_textures);
		item->setData(Qt::UserRole, static_cast<int>(texture));
		if (!plain) {
			images.append(texture);
			items.append(item);
		}
	}

	// Decoding large textures is slow: load them in the background and show each preview once ready.
	connect(&m_previewLoader, &QFutureWatcher<QImage>::resultReadyAt, this, [this, items](const int index) {
		const QImage preview = m_previewLoader.resultAt(index);
		if (preview.isNull()) {
			const bool wasCurrent = m_textures->currentItem() == items[index];
			delete items[index]; // Unreadable file: nothing to preview.
			if (wasCurrent) {
				selectTexture(boardStyle::Texture::Plain); // Qt would select the neighbour instead.
			}
			return;
		}
		items[index]->setIcon(untintedIcon(QPixmap::fromImage(preview)));
	});
	m_previewLoader.setFuture(QtConcurrent::mapped(images, [side, ratio](const boardStyle::Texture texture) {
		QImage preview = boardStyle::loadTexture(texture).scaled(side, side, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);
		preview.setDevicePixelRatio(ratio);
		return preview;
	}));
}

void BoardStyleDialog::selectTexture(const boardStyle::Texture texture) {
	for (int row = 0; row != m_textures->count(); ++row) {
		if (textureOf(*m_textures->item(row)) == texture) {
			m_textures->setCurrentRow(row);
			return;
		}
	}
	if (texture != boardStyle::Texture::Plain) {
		selectTexture(boardStyle::Texture::Plain); // Removed as unreadable: fall back to plain.
	}
}

} // namespace tengen::gui
