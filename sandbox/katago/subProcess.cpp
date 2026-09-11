#include "subProcess.hpp"

#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

bool SubProcess::start(const std::vector<std::string>& argv) {
	if (m_pid >= 0) {
		assert(false);
		return false; // Already running
	}

	if (argv.empty()) {
		return false;
	}

	// Setup Pipes
	if (pipe(m_inPipe) == -1 || pipe(m_outPipe) == -1) {
		return false;
	}

	// Subprocess
	pid_t pid = fork();
	if (pid < 0) {
		return false;
	}

	if (pid == 0) {
		// Child process

		// Redirect stderr to log file.
		int logFile = open("katago.log", O_WRONLY | O_CREAT | O_TRUNC, 0644);
		if (logFile == -1 || dup2(logFile, STDERR_FILENO) == -1) {
			// TODO: Handle stderr redirect. Child process will still function properly
		}
		close(logFile);

		// Redirect the pipe ends to stdin/stdout (copy)
		if (dup2(m_inPipe[0], STDIN_FILENO) == -1 || dup2(m_outPipe[1], STDOUT_FILENO) == -1) {
			perror("dup2 failed");
			_exit(EXIT_FAILURE);
		}

		// These are now redundant -> close them
		close(m_inPipe[0]);
		close(m_inPipe[1]);
		close(m_outPipe[0]);
		close(m_outPipe[1]);

		// Replace current process with the requested executable.
		std::vector<char*> childArgv;
		childArgv.reserve(argv.size() + 1);
		for (const std::string& arg: argv) {
			childArgv.push_back(const_cast<char*>(arg.c_str()));
		}
		childArgv.push_back(nullptr);
		execvp(childArgv[0], childArgv.data());

		// If  execvp command failed
		perror("execvp failed");
		_exit(EXIT_FAILURE); // Exit the subprocess with failure.
	} else {
		// Parent process
		m_pid = pid;

		// Close and reset unused pipes
		close(m_inPipe[0]);
		close(m_outPipe[1]);
		m_inPipe[0]  = -1;
		m_outPipe[1] = -1;
	}

	// TODO: Catch if the execvp failed?

	return true;
}

void SubProcess::stop() {
	// Cleanup pipes. Closing the child's stdin signals it to shut down.
	if (m_inPipe[0] >= 0) {
		close(m_inPipe[0]);
		m_inPipe[0] = -1;
	}
	if (m_inPipe[1] >= 0) {
		close(m_inPipe[1]);
		m_inPipe[1] = -1;
	}
	if (m_outPipe[0] >= 0) {
		close(m_outPipe[0]);
		m_outPipe[0] = -1;
	}
	if (m_outPipe[1] >= 0) {
		close(m_outPipe[1]);
		m_outPipe[1] = -1;
	}

	// Cleanup the child process.
	if (m_pid < 0) {
		return;
	}

	// Wait for process to shut down cleanly. Force kill after a timeout.
	constexpr auto shutdownTimeout = std::chrono::milliseconds(2000);
	constexpr auto pollInterval    = std::chrono::milliseconds(20);
	const auto deadline            = std::chrono::steady_clock::now() + shutdownTimeout;
	int status{0};
	pid_t result;
	while ((result = waitpid(m_pid, &status, WNOHANG)) == 0) {
		if (std::chrono::steady_clock::now() >= deadline) {
			kill(m_pid, SIGKILL);
			waitpid(m_pid, &status, 0); // SIGKILL can't be caught/blocked, returns promptly
			break;
		}
		std::this_thread::sleep_for(pollInterval);
	}
	m_pid = -1;
}

//! Send a command and wait for the response.
bool SubProcess::sendCommand(const std::string& command, std::string& response) {
	response = "";

	const std::string line = command + '\n';
	if (write(m_inPipe[1], line.c_str(), line.size()) == -1) {
		return false;
	}

	// Receive result
	char buffer[16];
	while (response.find("\n\n") == std::string::npos) {
		ssize_t n = read(m_outPipe[0], buffer, std::size(buffer));
		if (n <= 0) {
			break; // pipe closed or error
		}
		response.append(buffer, n);
	}

	//! Trim the KataGo response and evaluate if the command succeeded.
	const auto trimEval = [](std::string& response) -> bool {
		bool success = response[0] == '='; //!< Success if the response starts with '='.
		if (response.size() < 2) {
			return success;
		}

		response = response.substr(2); // Drop leading "= " or "? "

		// The response ends with \n\n
		const auto end = response.find_last_not_of(" \t\r\n");
		response       = (end == std::string::npos) ? std::string{} : response.substr(0, end + 1);
		return success;
	};

	return trimEval(response);
}
