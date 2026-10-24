#pragma once

#include "tengen/IAppSignal.hpp"
#include "tengen/sessionNetworkInfo.hpp"

namespace tengen::app {

class INetworkSession : public IAppSignalSource {
public:
	virtual ~INetworkSession() = default;

	virtual SessionNetworkInfo networkInfo() const = 0; //!< A copy, to read off the lock.
};

} // namespace tengen::app
