#pragma once

#include "tengen/INetworkSession.hpp"

#include <QObject>

namespace tengen {

namespace gui {
class GameWidget;
}

//! Shows what the server tells us about ourselves: our colour once welcomed, and why the connection ended.
//! \note Create it before connecting: it only shows what it is signalled.
class NetworkPresenter final : public QObject, public app::IAppSignalListener {
	Q_OBJECT

public:
	NetworkPresenter(app::INetworkSession& session, gui::GameWidget& gameWidget);
	~NetworkPresenter() override;

	void onAppEvent(app::AppSignalMask mask) override; //!< Signalled by session thread. Offload work to GUI thread.

private:
	void showConnection(const app::SessionNetworkInfo& info); //!< GUI thread only.

private:
	app::INetworkSession& m_session;
	gui::GameWidget& m_gameWidget;
};

} // namespace tengen
