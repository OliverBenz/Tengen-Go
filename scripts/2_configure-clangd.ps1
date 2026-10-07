# Configures the Win64-Ninja preset to generate compile_commands.json for clangd.
# Ninja with MSVC needs the compiler environment, so enter the x64 VS 2022 developer shell first.

$ErrorActionPreference = 'Stop'

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -version '[17.0,18.0)' -products * -latest -property installationPath
if (-not $vsPath) {
	throw 'Visual Studio 2022 installation not found.'
}

& "$vsPath\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
	throw 'cl.exe not found after entering the VS developer shell.'
}

# Always configure fresh: a configure outside the dev shell caches MinGW tools (e.g. Strawberry's ld.exe as linker).
Push-Location "$PSScriptRoot\.."
try {
	cmake --preset Win64-Ninja --fresh
	if ($LASTEXITCODE -ne 0) {
		throw "CMake configure failed with exit code $LASTEXITCODE."
	}

	# Copy to root so clangd finds it (CMake's symlink needs Developer Mode on Windows).
	cmake -E copy_if_different build/Win64-Ninja/compile_commands.json compile_commands.json
}
finally {
	Pop-Location
}
