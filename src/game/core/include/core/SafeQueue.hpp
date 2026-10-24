#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <utility>

namespace tengen {

//! Thread safe queue with a blocking Pop function.
template <class Entry>
class SafeQueue {
public:
	//! Push element onto the queue.
	void Push(const Entry& value);
	void Push(Entry&& value);

	//! Thread blocks here until there is an element to receive.
	Entry Pop();

	//! Returns true if the queue is empty; false otherwise.
	bool Empty() const;

protected:
	std::deque<Entry> m_queue;           //!< Stores the entries.
	mutable std::mutex m_mutex;          //!< Manage access to the queue.
	std::condition_variable m_condition; //!< Notify that element can be popped.
};


template <class Entry>
void SafeQueue<Entry>::Push(const Entry& value) {
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_queue.push_back(value);
	}
	m_condition.notify_one();
}

template <class Entry>
void SafeQueue<Entry>::Push(Entry&& value) {
	{
		std::lock_guard<std::mutex> lock(m_mutex);
		m_queue.push_back(std::move(value));
	}
	m_condition.notify_one();
}

template <class Entry>
Entry SafeQueue<Entry>::Pop() {
	std::unique_lock<std::mutex> lock(m_mutex);
	m_condition.wait(lock, [this] { return !m_queue.empty(); });

	Entry element = std::move(m_queue.front());
	m_queue.pop_front();
	return element;
}

template <class Entry>
bool SafeQueue<Entry>::Empty() const {
	std::lock_guard<std::mutex> lock(m_mutex);
	return m_queue.empty();
}

} // namespace tengen
