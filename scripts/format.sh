#!/bin/bash
# Formats every tracked C++ file with the repository's .clang-format.
#   ./format.sh          Format in place.
#   ./format.sh --check  Only report unformatted files. Fails if there are any.
# Set CLANG_FORMAT to use a specific binary. Different clang-format versions format slightly differently.

set -euo pipefail
cd "$(dirname "$0")/.."

clangFormat=${CLANG_FORMAT:-clang-format}

# Windows: fall back to the clang-format that ships with Visual Studio.
vswhere="/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe"
if ! command -v "$clangFormat" >/dev/null && [ -f "$vswhere" ]; then
	vsPath=$("$vswhere" -latest -products '*' -property installationPath | tr -d '\r')
	clangFormat="$(cygpath -u "$vsPath")/VC/Tools/Llvm/x64/bin/clang-format.exe"
fi

if ! command -v "$clangFormat" >/dev/null; then
	echo "clang-format not found. Install it or set CLANG_FORMAT." >&2
	exit 1
fi

mode=(-i)
if [ "${1:-}" = "--check" ]; then
	mode=(--dry-run -Werror --ferror-limit=1)
fi

"$clangFormat" --version
git ls-files -z '*.cpp' '*.hpp' '*.h' | xargs -0 "$clangFormat" --style=file "${mode[@]}"
