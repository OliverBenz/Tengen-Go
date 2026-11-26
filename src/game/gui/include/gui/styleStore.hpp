#pragma once

#include "gui/resources.hpp"

#include <QObject>

namespace tengen::gui {

//! Holds the one shared Style and announces changes via styleChanged.
//! Widgets usually don't hold onto this store; it's wired up externally instead.
class StyleStore : public QObject {
	Q_OBJECT

public:
	explicit StyleStore(Style style = Style{}, QObject* parent = nullptr);

	const Style& style() const;        //!< The style currently in use.
	void setStyle(const Style& style); //!< Set a new style; does nothing if it's unchanged.

signals:
	void styleChanged(const Style& style);

private:
	Style m_style;
};

} // namespace tengen::gui
