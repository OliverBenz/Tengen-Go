#include "gui/styleStore.hpp"

namespace tengen::gui {

StyleStore::StyleStore(Style style, QObject* parent)
    : QObject(parent), m_style(style) {
}

const Style& StyleStore::style() const {
	return m_style;
}

void StyleStore::setStyle(const Style& style) {
	if (style == m_style) {
		return;
	}
	m_style = style;
	emit styleChanged(m_style);
}

} // namespace tengen::gui
