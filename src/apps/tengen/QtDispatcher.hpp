#pragma once

#include "tengen/IAppSignal.hpp"

#include <QObject>
#include <qobjectdefs.h>

namespace tengen {

class QtDispatcher : public QObject, public app::IDispatcher {
	Q_OBJECT;

public:
	void post(std::function<void()> callFunction) const override; //!< Queue a new update function to be executed.
	void flush() const;                                           //!< Handle all queued update calls right now.
};

} // namespace tengen
