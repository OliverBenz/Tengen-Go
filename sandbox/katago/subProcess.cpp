#include "subProcess.hpp"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

//! Close a file descriptor and mark it as closed.
void closeFd(int& fd) {
	if (fd >= 0) {
		close(fd);
		fd = -1;
	}
}

//! Write the whole buffer. A pipe may take the data in several chunks.
bool writeAll(const int fd, const std::string& data) {
	std::size_t written = 0u;
	while (written < data.size()) {
		const ssize_t count = write(fd, data.data() + written, data.size() - written);
		if (count < 0) {
			if (errno == EINTR) {
				continue; // Interrupted before anything went out. Retry.
			}
			return false;
		}
		written += static_cast<std::size_t>(count);
	}
	return true;
}

} // namespace


SubProcess::~SubProcess() {
	stop();
}

bool SubProcess::start(const std::vector<std::string>& argv, const std::string& logFile) {
	if (isRunning()) {
		assert(false);
		return false; // Already running
	}

	if (argv.empty()) {
		return false;
	}

	// Lay the argument list out before forking. Between fork and exec the child may only touch
	// memory that is already there, so nothing in it is allowed to allocate.
	std::vector<char*> childArgv;
	childArgv.reserve(argv.size() + 1u);
	for (const std::string& arg: argv) {
		childArgv.push_back(const_cast<char*>(arg.c_str()));
	}
	childArgv.push_back(nullptr);

	// Setup Pipes
	if (pipe(m_inPipe) == -1) {
		return false;
	}
	if (pipe(m_outPipe) == -1) {
		closePipes();
		return false;
	}

	// A child that died must not take us down with it. write() reports EPIPE instead.
	std::signal(SIGPIPE, SIG_IGN);

	// Subprocess
	const pid_t pid = fork();
	if (pid < 0) {
		closePipes();
		return false;
	}

	if (pid == 0) {
		execChild(childArgv.data(), logFile.c_str()); // Does not return.
	}

	// Parent process
	m_pid = pid;

	// The other ends of the pipes belong to the child now.
	closeFd(m_inPipe[0]);
	closeFd(m_outPipe[1]);
	return true;
}

void SubProcess::stop() {
	// Closing the child's stdin asks it to shut down. Its own exit then closes the other pipe from
	// the far side, which is what releases a readUntil() that is still blocking.
	closeFd(m_inPipe[1]);

	waitForExit();
	closePipes();
}

bool SubProcess::isRunning() const {
	return m_pid >= 0;
}

bool SubProcess::sendLine(const std::string& line) {
	if (m_inPipe[1] < 0) {
		return false;
	}

	return writeAll(m_inPipe[1], line + '\n');
}

bool SubProcess::readUntil(std::string& data, const std::string_view terminator) {
	data.clear();
	if (m_outPipe[0] < 0) {
		return false;
	}

	// The child answers one request at a time, so nothing of the next answer can trail the terminator.
	char buffer[512];
	while (data.find(terminator) == std::string::npos) {
		const ssize_t count = read(m_outPipe[0], buffer, std::size(buffer));
		if (count < 0 && errno == EINTR) {
			continue; // Interrupted before anything arrived. Retry.
		}
		if (count <= 0) {
			return false; // Pipe closed or error. Whatever we hold is incomplete.
		}
		data.append(buffer, static_cast<std::size_t>(count));
	}
	return true;
}

void SubProcess::execChild(char* const argv[], const char* logFile) {
	std::signal(SIGPIPE, SIG_DFL); // The parent ignores it, the child keeps the default.

	// Redirect stderr to the log file. The child still works without it.
	const int logFd = open(logFile, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (logFd != -1) {
		dup2(logFd, STDERR_FILENO);
		close(logFd);
	}

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
	execvp(argv[0], argv);

	// Only reached if the executable could not be launched. The closed pipe tells the parent.
	perror("execvp failed");
	_exit(EXIT_FAILURE);
}

void SubProcess::waitForExit() {
	if (m_pid < 0) {
		return;
	}

	// Wait for process to shut down cleanly. Force kill after a timeout.
	constexpr auto shutdownTimeout = std::chrono::milliseconds(2000);
	constexpr auto pollInterval    = std::chrono::milliseconds(20);
	const auto deadline            = std::chrono::steady_clock::now() + shutdownTimeout;

	int status{0};
	while (waitpid(m_pid, &status, WNOHANG) == 0) {
		if (std::chrono::steady_clock::now() >= deadline) {
			kill(m_pid, SIGKILL);
			waitpid(m_pid, &status, 0); // SIGKILL can't be caught/blocked, returns promptly
			break;
		}
		std::this_thread::sleep_for(pollInterval);
	}
	m_pid = -1;
}

void SubProcess::closePipes() {
	closeFd(m_inPipe[0]);
	closeFd(m_inPipe[1]);
	closeFd(m_outPipe[0]);
	closeFd(m_outPipe[1]);
}
