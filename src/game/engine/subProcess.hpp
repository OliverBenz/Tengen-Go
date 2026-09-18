#pragma once

#include <string>
#include <string_view>
#include <sys/types.h>
#include <vector>

namespace tengen::engine {

//! Runs a child process and talks to it over its stdin/stdout.
//! The transport is line oriented and knows nothing about the protocol spoken on top of it.
class SubProcess {
public:
	SubProcess()                             = default;
	SubProcess(const SubProcess&)            = delete;
	SubProcess& operator=(const SubProcess&) = delete;
	~SubProcess();

	//! Launch the child. argv[0] is the executable, logFile takes over its stderr.
	bool start(const std::vector<std::string>& argv, const std::string& logFile);
	void stop();            //!< Shut the child down and reap it. Releases a readUntil() that is still blocking.
	bool isRunning() const; //!< True while a child process is attached.

	bool sendLine(const std::string& line);                         //!< Write one line to the child's stdin.
	bool readUntil(std::string& data, std::string_view terminator); //!< Read the child's stdout up to and including the terminator.

private:
	void execChild(char* const argv[], const char* logFile); //!< Runs in the forked child. Only returns if the launch failed.
	void waitForExit();                                      //!< Wait for the child to exit. Force kill it after a timeout.
	void closePipes();                                       //!< Close every pipe end we still own.

private:
	pid_t m_pid{-1};          //!< Child process Id.
	int m_inPipe[2]{-1, -1};  //!< Pipe: parent -> child
	int m_outPipe[2]{-1, -1}; //!< Pipe: child  -> parent
};

} // namespace tengen::engine
