#pragma once

#include "gui/resources.hpp"

class QSettings;

namespace tengen::gui {

//! Style stored in the settings. Missing or unknown values fall back to the default style.
Style loadStyle(const QSettings& settings);

//! Store the style in the settings.
void saveStyle(QSettings& settings, const Style& style);

} // namespace tengen::gui
