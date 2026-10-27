#pragma once

#include <cstdint>

namespace tengen::app {

//! Types of signals.
enum AppSignal : uint64_t {
	AS_None             = 0,
	AS_BoardChange      = 1ull << 0u, //!< Board was modified (also sent on reset. Dont treat as move).
	AS_PlayerChange     = 1ull << 1u, //!< Active player changed.
	AS_StateChange      = 1ull << 2u, //!< Game state changed. Started or finished.
	AS_StonePlaced      = 1ull << 3u, //!< A move put a stone on the board.
	AS_ConnectionChange = 1ull << 4u, //!< Connecting, welcomed or ended.
	AS_NewChat          = 1ull << 5u, //!< New chat message received.
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
