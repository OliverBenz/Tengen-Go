#pragma once

#include <cstdint>

namespace tengen::app {

//! Types of signals.
enum AppSignal : uint64_t {
	AS_None         = 0,
	AS_BoardChange  = 1 << 0, //!< Board was modified (also sent on reset. Dont treat as move).
	AS_PlayerChange = 1 << 1, //!< Active player changed.
	AS_StateChange  = 1 << 2, //!< Game state changed. Started or finished.
	AS_NewChat      = 1 << 3, //!< New chat message received.
	AS_StonePlaced  = 1 << 4, //!< A move put a stone on the board.
};

class IAppSignalListener {
public:
	virtual ~IAppSignalListener() = default;

	virtual void onAppEvent(AppSignal signal) = 0;
};

class IAppSignalSource {
public:
	virtual ~IAppSignalSource() = default;

	virtual void subscribe(app::IAppSignalListener* listener, uint64_t mask) = 0;
	virtual void unsubscribe(app::IAppSignalListener* listener)              = 0;
};

} // namespace tengen::app
