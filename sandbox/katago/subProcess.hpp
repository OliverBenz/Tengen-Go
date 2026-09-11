#pragma once

#include <string>
#include <sys/types.h>
#include <vector>

class SubProcess {
public:
	SubProcess() = default;

	bool start(const std::vector<std::string>& argv);
	void stop();
	bool sendCommand(const std::string& command, std::string& response); //!< Send a command and wait for the response.

private:
	pid_t m_pid{-1};          //!< Child process Id.
	int m_inPipe[2]{-1, -1};  //!< Pipe: parent -> child
	int m_outPipe[2]{-1, -1}; //!< Pipe: child  -> parent
};
