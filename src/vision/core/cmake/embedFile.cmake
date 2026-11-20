# Writes OUTPUT: a C++ source whose FUNCTION() returns the bytes of INPUT.

foreach(arg INPUT OUTPUT HEADER NAMESPACE FUNCTION)
	if(NOT DEFINED ${arg})
		message(FATAL_ERROR "embedFile.cmake: ${arg} is not set.")
	endif()
endforeach()

file(READ "${INPUT}" hex HEX)
string(LENGTH "${hex}" hexLength)
math(EXPR size "${hexLength} / 2")

# Git LFS pointer: "version https://git-lfs".
if(hex MATCHES "^76657273696f6e2068747470733a2f2f6769742d6c6673")
	message(FATAL_ERROR "${INPUT} is a Git LFS pointer, not the file. Run scripts/1_fetch.sh first.")
endif()

string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," bytes "${hex}")
get_filename_component(name "${INPUT}" NAME)

file(WRITE "${OUTPUT}.tmp"
"// Generated from ${name} by embedFile.cmake. Do not edit.
#include \"${HEADER}\"

#include <array>

namespace ${NAMESPACE} {

static constexpr std::array<unsigned char, ${size}> DATA{${bytes}};

std::span<const unsigned char> ${FUNCTION}() {
	return DATA;
}

} // namespace ${NAMESPACE}
")
file(COPY_FILE "${OUTPUT}.tmp" "${OUTPUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUTPUT}.tmp")
