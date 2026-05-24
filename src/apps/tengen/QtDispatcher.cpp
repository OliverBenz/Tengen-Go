#include "QtDispatcher.hpp"

#include <QCoreApplication>
#include <QEvent>

namespace tengen {

void QtDispatcher::post(std::function<void()> callFunction) const {
	QMetaObject::invokeMethod(const_cast<QtDispatcher*>(this), [fn = std::move(callFunction)]() { fn(); }, Qt::QueuedConnection);
}

void QtDispatcher::flush() const {
	// Process all queued MetaCall events now.
	QCoreApplication::sendPostedEvents(const_cast<QtDispatcher*>(this), QEvent::MetaCall);
}

} // namespace tengen
