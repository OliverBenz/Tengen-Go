#pragma once

#include "model/player.hpp"

#include <QDialog>

class QComboBox;
class QLabel;
class QSlider;

namespace tengen::gui {

class BotDialog : public QDialog {
	Q_OBJECT

public:
	explicit BotDialog(QWidget* parent = nullptr);

	unsigned boardSize() const;
	Skill skill() const; //!< Rank the bot should play at.
	bool humanPlaysBlack() const;

private:
	QComboBox* m_boardSize{nullptr};
	QSlider* m_skill{nullptr};
	QLabel* m_skillLabel{nullptr}; //!< Shows the rank the slider currently sits on.
	QComboBox* m_colour{nullptr};
};

} // namespace tengen::gui
