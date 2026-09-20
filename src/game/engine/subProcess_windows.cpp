#include "subProcess.hpp"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace tengen::engine {
namespace {

constexpr std::size_t readEnd  = 0u; //!< Index of the end of a pipe that is read from.
constexpr std::size_t writeEnd = 1u; //!< Index of the end of a pipe that is written to.

//! Close a handle and mark it as closed.
void closeHandle(HANDLE& handle) {
	if (handle != nullptr) {
		CloseHandle(handle);
		handle = nullptr;
	}
}

//! Write the whole buffer. A pipe may take the data in several chunks.
bool writeAll(const HANDLE handle, const std::string& data) {
	std::size_t written = 0u;
	while (written < data.size()) {
		DWORD count = 0u;
		if (!WriteFile(handle, data.data() + written, static_cast<DWORD>(data.size() - written), &count, nullptr) || count == 0u) {
			return false;
		}
		written += count;
	}
	return true;
}

//! Create a pipe and keep our own end out of the child's hands.
//! \param parentEnd Index of the end the parent keeps. The child inherits the other one.
bool createPipe(HANDLE (&pipeEnds)[2], const std::size_t parentEnd) {
	SECURITY_ATTRIBUTES attributes{};
	attributes.nLength        = sizeof(attributes);
	attributes.bInheritHandle = TRUE; // Both ends start inheritable. The parent's end is cleared below.

	if (!CreatePipe(&pipeEnds[readEnd], &pipeEnds[writeEnd], &attributes, 0)) {
		pipeEnds[readEnd]  = nullptr;
		pipeEnds[writeEnd] = nullptr;
		return false;
	}

	// An inherited copy would hold the pipe open in the child, so the EOF on shutdown never arrives.
	return SetHandleInformation(pipeEnds[parentEnd], HANDLE_FLAG_INHERIT, 0) != 0;
}

//! Append one argument the way CommandLineToArgv() parses it back: quote it when it carries a
//! separator and double up every run of backslashes that ends on a quote.
void appendArgument(std::string& commandLine, const std::string& argument) {
	if (!argument.empty() && argument.find_first_of(" \t\"") == std::string::npos) {
		commandLine += argument;
		return;
	}

	commandLine += '"';
	for (std::size_t i = 0u; i < argument.size(); ++i) {
		std::size_t backslashes = 0u;
		while (i < argument.size() && argument[i] == '\\') {
			++backslashes;
			++i;
		}

		if (i == argument.size()) {
			commandLine.append(backslashes * 2u, '\\'); // Run in front of the closing quote.
			break;
		}
		if (argument[i] == '"') {
			commandLine.append(backslashes * 2u + 1u, '\\'); // Run in front of a quote we escape.
		} else {
			commandLine.append(backslashes, '\\');
		}
		commandLine += argument[i];
	}
	commandLine += '"';
}

//! Build the single command line CreateProcess() takes out of the argument list.
std::string buildCommandLine(const std::vector<std::string>& argv) {
	std::string commandLine;
	for (const std::string& argument: argv) {
		if (!commandLine.empty()) {
			commandLine += ' ';
		}
		appendArgument(commandLine, argument);
	}
	return commandLine;
}

} // namespace


class SubProcess::Pimpl {
public:
	HANDLE m_process{nullptr};             //!< Child process handle.
	HANDLE m_inPipe[2]{nullptr, nullptr};  //!< Pipe: parent -> child
	HANDLE m_outPipe[2]{nullptr, nullptr}; //!< Pipe: child  -> parent
};

SubProcess::SubProcess()
    : m_pimpl{std::make_unique<SubProcess::Pimpl>()} {
}

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

	// Arm the flag for this launch. Set before the child exists, so a stop() racing the launch below
	// still latches and is caught after the process is created.
	m_stopped = false;

	// Setup Pipes
	if (!createPipe(m_pimpl->m_inPipe, writeEnd) || !createPipe(m_pimpl->m_outPipe, readEnd)) {
		closePipes();
		return false;
	}

	// Redirect stderr to the log file. The child still works without it and keeps ours instead.
	SECURITY_ATTRIBUTES logAttributes{};
	logAttributes.nLength        = sizeof(logAttributes);
	logAttributes.bInheritHandle = TRUE;

	HANDLE logHandle = CreateFileA(logFile.c_str(), GENERIC_WRITE, FILE_SHARE_READ, &logAttributes, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (logHandle == INVALID_HANDLE_VALUE) {
		logHandle = nullptr;
	}

	// The child takes its pipe ends through the standard handles instead of a fork.
	STARTUPINFOA startupInfo{};
	startupInfo.cb         = sizeof(startupInfo);
	startupInfo.dwFlags    = STARTF_USESTDHANDLES;
	startupInfo.hStdInput  = m_pimpl->m_inPipe[readEnd];
	startupInfo.hStdOutput = m_pimpl->m_outPipe[writeEnd];
	startupInfo.hStdError  = logHandle != nullptr ? logHandle : GetStdHandle(STD_ERROR_HANDLE);

	// Subprocess. CreateProcess() writes into the command line, so it gets our own buffer.
	std::string commandLine = buildCommandLine(argv);
	PROCESS_INFORMATION processInfo{};
	const bool started = CreateProcessA(nullptr, commandLine.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &startupInfo, &processInfo) != 0;

	closeHandle(logHandle); // The child holds its own copy now.

	// Unlike fork(), this already fails when the executable cannot be launched.
	if (!started) {
		closePipes();
		return false;
	}

	// Parent process
	CloseHandle(processInfo.hThread); // We only ever wait on the process itself.
	m_pimpl->m_process = processInfo.hProcess;

	// The other ends of the pipes belong to the child now.
	closeHandle(m_pimpl->m_inPipe[readEnd]);
	closeHandle(m_pimpl->m_outPipe[writeEnd]);

	// A stop that arrived while we were launching only reaches the child here. It has not spoken
	// yet, so it goes down right away instead of getting the grace period stop() would give it.
	if (m_stopped) {
		TerminateProcess(m_pimpl->m_process, EXIT_FAILURE);
		stop();
		return false;
	}
	return true;
}

void SubProcess::stop() {
	m_stopped = true;

	// Closing the child's stdin asks it to shut down. Its own exit then closes the other pipe from
	// the far side, which is what releases a readUntil() that is still blocking.
	closeHandle(m_pimpl->m_inPipe[writeEnd]);

	waitForExit();
	closePipes();
}

bool SubProcess::isRunning() const {
	return m_pimpl->m_process != nullptr;
}

bool SubProcess::sendLine(const std::string& line) {
	if (m_pimpl->m_inPipe[writeEnd] == nullptr) {
		return false;
	}

	return writeAll(m_pimpl->m_inPipe[writeEnd], line + '\n');
}

bool SubProcess::readUntil(std::string& data, const std::string_view terminator) {
	data.clear();
	if (m_pimpl->m_outPipe[readEnd] == nullptr) {
		return false;
	}

	// The child answers one request at a time, so nothing of the next answer can trail the terminator.
	char buffer[512];
	while (data.find(terminator) == std::string::npos) {
		DWORD count = 0u;
		if (!ReadFile(m_pimpl->m_outPipe[readEnd], buffer, static_cast<DWORD>(std::size(buffer)), &count, nullptr) || count == 0u) {
			return false; // Pipe closed or error. Whatever we hold is incomplete.
		}
		data.append(buffer, count);
	}
	return true;
}

void SubProcess::waitForExit() {
	if (m_pimpl->m_process == nullptr) {
		return;
	}

	// Wait for process to shut down cleanly. Force kill after a timeout.
	constexpr DWORD shutdownTimeout = 2000u;
	if (WaitForSingleObject(m_pimpl->m_process, shutdownTimeout) == WAIT_TIMEOUT) {
		TerminateProcess(m_pimpl->m_process, EXIT_FAILURE);
		WaitForSingleObject(m_pimpl->m_process, INFINITE); // TerminateProcess can't be blocked, returns promptly
	}

	closeHandle(m_pimpl->m_process);
}

void SubProcess::closePipes() {
	closeHandle(m_pimpl->m_inPipe[readEnd]);
	closeHandle(m_pimpl->m_inPipe[writeEnd]);
	closeHandle(m_pimpl->m_outPipe[readEnd]);
	closeHandle(m_pimpl->m_outPipe[writeEnd]);
}

} // namespace tengen::engine
