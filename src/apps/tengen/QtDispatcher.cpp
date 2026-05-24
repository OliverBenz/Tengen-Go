#include "QtDispatcher.hpp"

namespace tengen {

void QtDispatcher::post(std::function<void()> callFunction) const {
	QMetaObject::invokeMethod(const_cast<QtDispatcher*>(this), [fn = std::move(callFunction)]() { fn(); }, Qt::QueuedConnection);
}

} // namespace tengen
