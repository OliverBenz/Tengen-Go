#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace tengen::engine {

//! Runs a child process and talks to it over its stdin/stdout.
//! The transport is line oriented and knows nothing about the protocol spoken on top of it.
class SubProcess {
public:
	SubProcess();
	SubProcess(const SubProcess&)            = delete;
	SubProcess& operator=(const SubProcess&) = delete;
	~SubProcess();

	//! Launch the child. argv[0] is the executable, logFile takes over its stderr.
	//! Fails when stop() came in first: a launch is never left running behind its own shutdown.
	bool start(const std::vector<std::string>& argv, const std::string& logFile);
	void stop();            //!< Shut the child down and reap it. Releases a readUntil() that is still blocking.
	bool isRunning() const; //!< True while a child process is attached.

	bool sendLine(const std::string& line);                         //!< Write one line to the child's stdin.
	bool readUntil(std::string& data, std::string_view terminator); //!< Read the child's stdout up to and including the terminator.

private:
	void waitForExit(); //!< Wait for the child to exit. Force kill it after a timeout.
	void closePipes();  //!< Close every pipe end we still own.

private:
	class Pimpl;
	std::unique_ptr<Pimpl> m_pimpl{nullptr}; //!< Implementation pointer. Allows to select between windows and linux implementation.
	std::atomic<bool> m_stopped{false};      //!< Set by stop(), read by a start() that is still running. Never cleared.
};

} // namespace tengen::engine
