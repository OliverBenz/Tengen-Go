#pragma once

#include "BoardPresenter.hpp"
#include "ChatPresenter.hpp"
#include "gui/gameWidget.hpp"
#include "tengen/IChatSession.hpp"
#include "tengen/IGameSession.hpp"

#include <QObject>

#include <memory>

namespace tengen {

class GamePresenter : public QObject, public app::IAppSignalListener {
	Q_OBJECT

public:
	GamePresenter(app::IGameSession& game, gui::GameWidget& gameWidget);
	~GamePresenter() override;

	void addChatWindow(app::IChatSession& chat);

	void onAppEvent(app::AppSignal signal) override; //!< Signalled by session thread. Offload work to GUI thread.

private slots:
	void onPassRequested();   //!< Handle pass event from Board Widget.
	void onResignRequested(); //!< Handle resign event from Board Widget.

private:
	void showStatus(); //!< Show the session's status and whose move it is.

private:
	app::IGameSession& m_game;
	gui::GameWidget& m_gameWidget;
	std::unique_ptr<BoardPresenter> m_boardPresenter = nullptr;
	std::unique_ptr<ChatPresenter> m_chatPresenter   = nullptr;
};

} // namespace tengen
