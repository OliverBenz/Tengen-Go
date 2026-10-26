#!/bin/bash
# Runs clang-tidy with the repository's .clang-tidy over our own sources.
#   ./tidy.sh            Check everything in src/ and tests/.
#   ./tidy.sh src/net    Only check files below this path.
# Uses the compile database of a configured build. Set BUILD_DIR to pick another one.

set -euo pipefail
cd "$(dirname "$0")/.."

buildDir=${BUILD_DIR:-build/Linux64-Debug}
filter=${1:-src|tests}

if [ ! -f "$buildDir/compile_commands.json" ]; then
	echo "No compile database in $buildDir. Configure that build first, or set BUILD_DIR." >&2
	exit 1
fi

# The database holds GCC's flags. Clang rejects Qt's -mno-direct-extern-access, so drop it from a copy.
tidyDir="$buildDir/clang-tidy"
mkdir -p "$tidyDir"
sed 's/ -mno-direct-extern-access//g' "$buildDir/compile_commands.json" >"$tidyDir/compile_commands.json"

# GCC-only warning flags are unknown to clang; compiler warnings aren't clang-tidy's business here.
run-clang-tidy -p "$tidyDir" -quiet \
	-extra-arg=-Wno-unknown-warning-option \
	-extra-arg=-Wno-error \
	"^$(pwd)/($filter)/"
