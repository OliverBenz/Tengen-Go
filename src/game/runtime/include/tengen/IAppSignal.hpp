#pragma once

#include <cstdint>
#include <functional>

namespace tengen::app {

using AppSignalMask = std::uint64_t;

//! Types of signals.
enum AppSignal : AppSignalMask {
	AS_None         = 0,
	AS_BoardChange  = 1 << 0, //!< Board was modified.
	AS_PlayerChange = 1 << 1, //!< Active player changed.
	AS_StateChange  = 1 << 2, //!< Game state changed. Started or finished.
	AS_NewChat      = 1 << 3, //!< New chat message received.
};

//! Every class who listens to application signals.
class IAppSignalListener {
public:
	virtual ~IAppSignalListener() = default;

	//! Function applying application signal updates.
	//! \param [in] signalMask Mask of appsignals with each bit signalling one AppSignal state which has been updated.
	virtual void onAppEvent(AppSignalMask signalMask) = 0;
};

//! Every class who dispatches application signals.
class IAppSignalSource {
public:
	virtual ~IAppSignalSource() = default;

	virtual void subscribe(app::IAppSignalListener* listener, AppSignalMask mask) = 0;
	virtual void unsubscribe(app::IAppSignalListener* listener)                   = 0;
};

//! Helper class which asynchronously calls a function through post.
class IDispatcher {
public:
	virtual ~IDispatcher() = default;

	//! Asynchronously dispatch a function call to 'callFunction'.
	virtual void post(std::function<void()> callFunction) const = 0;
};

} // namespace tengen::app
