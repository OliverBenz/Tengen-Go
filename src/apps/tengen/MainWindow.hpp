#pragma once

#include "engine/engineCatalog.hpp"

#include <QCloseEvent>
#include <QMainWindow>
#include <QString>

namespace tengen::gui {

class GameWidget;

class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget* parent = nullptr);
	~MainWindow() override;

	GameWidget& gameWidget();

	//! Let the user set a bot game up against one of the engines. Answers with gameBotRequested() once accepted.
	void openBotDialog(const engine::InstalledEngines& engines);

signals:
	void gameLocalRequested();
	void botDialogRequested(); //!< The user wants a bot game. Answer with openBotDialog().
	void gameBotRequested(unsigned boardSize, const engine::EngineConfig& engineConfig, bool humanPlaysBlack);
	void connectRequested(const QString& hostIp);
	void hostRequested(unsigned boardSize);
	void shutdownRequested();

private:
	//! Initial setup constructing the layout of the window.
	void buildLayout();

private:
	void openConnectDialog();
	void openHostDialog();
	void openRulesDialog();

protected:
	void closeEvent(QCloseEvent* event) override;

private:
	GameWidget* m_gameWidget = nullptr;
};

} // namespace tengen::gui
