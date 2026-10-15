#!/usr/bin/env python3
"""Install the exact Qt and OpenCV versions that CMakeLists.txt requires.

Everything goes into a shared folder outside the repository, so deleting build/ never forces a reinstall.
Each version gets its own subfolder, so a version bump installs next to the old one.
Running the script again skips whatever is already installed. Delete a version's folder to reinstall it.

Qt:     Official prebuilt binaries, downloaded with aqtinstall (installed into a private venv).
OpenCV: Official prebuilt release on Windows. Built from source on Linux (no official binaries).

Usage:  python scripts/setup_deps.py [--deps-dir DIR] [--qt-arch ARCH] [--skip-opencv]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# aqtinstall 3.3.0 (latest release) can't read the Windows repository layout Qt uses since 6.11.
# The fix (miurahr/aqtinstall#1000) is only on master, so pin a master commit until a release has it.
AQT_REQUIREMENT = "aqtinstall @ https://github.com/miurahr/aqtinstall/archive/076e1659807d0b362a3ed684d54c2e9c775eb9c7.zip"

QT_MODULES = ["qtmultimedia"]  # Add-on modules. Widgets and Concurrent are part of the base package.
QT_HOST = {"Windows": "windows", "Linux": "linux"}
QT_DEFAULT_ARCH = {"Windows": "win64_msvc2022_64", "Linux": "linux_gcc_64"}
CONFIGURE_PRESET = {"Windows": "Win64", "Linux": "Linux64-Debug"}

STAMP = ".tengen-installed"  # Written last, so an interrupted install is never mistaken for a finished one.


@dataclass
class Versions:
    qt: str
    opencv: str


# ----- Setup -----

def check_platform() -> str:
    """Return the OS name, or exit if it is not x64 Windows or Linux."""
    system = platform.system()
    if system not in QT_HOST or platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit(f"error: only x64 Windows and Linux are supported (this is {system} {platform.machine()}).")
    return system


def default_deps_dir(system: str) -> Path:
    """Shared install folder in the user's data folder."""
    if system == "Windows":
        return Path(os.environ["LOCALAPPDATA"]) / "tengen" / "deps"
    data_home = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(data_home) / "tengen" / "deps"


def parse_args(system: str) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Install the exact Qt and OpenCV versions Tengen requires.")
    parser.add_argument("--deps-dir", type=Path, default=os.environ.get("TENGEN_DEPS_DIR") or default_deps_dir(system),
                        help="Install folder. Default: TENGEN_DEPS_DIR, else the user's data folder.")
    parser.add_argument("--qt-arch", default=QT_DEFAULT_ARCH[system], help="aqt architecture name.")
    parser.add_argument("--skip-opencv", action="store_true",
                        help="Only install Qt. Configuring still needs OpenCV until vision can be turned off.")
    return parser.parse_args()


def read_versions() -> Versions:
    """Read the pinned versions from the top-level CMakeLists.txt, the single place they are defined."""
    text = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    def version(name: str) -> str:
        match = re.search(rf"^\s*set\(\s*{name}\s+([\d.]+)\s*\)", text, re.MULTILINE)
        if not match:
            sys.exit(f"error: {name} not found in CMakeLists.txt")
        return match.group(1)

    return Versions(qt=version("TENGEN_QT_VERSION"), opencv=version("TENGEN_OPENCV_VERSION"))


# ----- Helpers -----

def run(cmd: list[str | Path]) -> None:
    """Print and run a command. Raises CalledProcessError on failure."""
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def is_installed(folder: Path, content: str = "") -> bool:
    """Whether the folder holds a finished install with the given stamp content."""
    stamp = folder / STAMP
    return stamp.is_file() and stamp.read_text(encoding="utf-8") == content


def mark_installed(folder: Path, content: str = "") -> None:
    """Mark the folder as a finished install."""
    (folder / STAMP).write_text(content, encoding="utf-8")


def find_config(root: Path, file_name: str) -> Path:
    """Folder of the shallowest <file_name> below root (e.g. the top-level OpenCVConfig.cmake)."""
    matches = sorted(root.rglob(file_name), key=lambda p: len(p.parts))
    if not matches:
        sys.exit(f"error: no {file_name} found in {root}")
    return matches[0].parent


def open_url(url: str):
    """Open a URL. GitHub rejects requests without a User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "tengen-setup"})
    return urllib.request.urlopen(request)


def download(url: str, target: Path) -> None:
    """Download to a .part file first, so a cut-off download never looks complete."""
    print(f"Downloading {url}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with open_url(url) as response, open(part, "wb") as file:
        shutil.copyfileobj(response, file)
    part.replace(target)


# ----- Qt -----

def ensure_aqt(deps_dir: Path) -> Path:
    """Return the Python of a private venv with the pinned aqtinstall. Rebuilds the venv when the pin changes."""
    venv = deps_dir / "tools" / "aqt-venv"
    python = venv / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")

    if not is_installed(venv, AQT_REQUIREMENT):
        shutil.rmtree(venv, ignore_errors=True)
        run([sys.executable, "-m", "venv", venv])
        run([python, "-m", "pip", "install", "--quiet", "--upgrade", "pip", AQT_REQUIREMENT])
        mark_installed(venv, AQT_REQUIREMENT)
    return python


def download_qt(python: Path, version: str, arch: str, out_dir: Path) -> None:
    """Install Qt with aqt into <out_dir>/<version>/."""
    host = QT_HOST[platform.system()]
    try:
        run([python, "-m", "aqt", "install-qt", host, "desktop", version, arch,
             "-m", *QT_MODULES, "--outputdir", out_dir])
    except subprocess.CalledProcessError:
        sys.exit(f"error: Qt install failed, see the aqt output above. If the architecture is wrong, list them with:\n"
                 f"  {python} -m aqt list-qt {host} desktop --arch {version}\n"
                 f"and pass one with --qt-arch.")


def install_qt(deps_dir: Path, version: str, arch: str) -> Path:
    """Install Qt unless already there. Return the folder that holds Qt6Config.cmake."""
    out_dir = deps_dir / "qt" / arch  # One folder per arch, so each version folder holds exactly one arch.
    version_dir = out_dir / version
    modules = " ".join(QT_MODULES)

    if is_installed(version_dir, modules):
        print(f"Qt {version} already installed: {version_dir}")
    else:
        download_qt(ensure_aqt(deps_dir), version, arch, out_dir)
        mark_installed(version_dir, modules)

    return find_config(version_dir, "Qt6Config.cmake")


# ----- OpenCV Windows -----

def find_opencv_windows_asset(version: str) -> tuple[str, str]:
    """Look up name and URL of the official Windows package, instead of guessing the name."""
    with open_url(f"https://api.github.com/repos/opencv/opencv/releases/tags/{version}") as response:
        assets = json.load(response)["assets"]

    asset = next((a for a in assets if a["name"].endswith("-windows.exe")), None)
    if asset is None:
        sys.exit(f"error: OpenCV release {version} has no Windows package.")
    return asset["name"], asset["browser_download_url"]


def extract_opencv_windows(installer: Path, target: Path) -> None:
    """Extract the self-extracting 7z package into target."""
    shutil.rmtree(target, ignore_errors=True)  # Leftovers of an interrupted extraction.
    run([installer, f"-o{target}", "-y"])


def install_opencv_windows(deps_dir: Path, version: str) -> Path:
    """Install OpenCV unless already there. Return the folder that holds the top-level OpenCVConfig.cmake."""
    target = deps_dir / f"opencv-{version}"

    if is_installed(target):
        print(f"OpenCV {version} already installed: {target}")
    else:
        name, url = find_opencv_windows_asset(version)
        installer = deps_dir / "downloads" / name
        download(url, installer)
        extract_opencv_windows(installer, target)
        installer.unlink()
        mark_installed(target)

    return find_config(target, "OpenCVConfig.cmake")


# ----- OpenCV Linux -----

def clone_opencv(version: str, src_dir: Path) -> None:
    """Shallow clone of the release tag."""
    run(["git", "clone", "--depth", "1", "--branch", version, "https://github.com/opencv/opencv.git", src_dir])


def build_opencv(src_dir: Path, build_dir: Path, prefix: Path) -> None:
    """Configure, build and install a Release build without tests, examples, Java and Python."""
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    run(["cmake", "-S", src_dir, "-B", build_dir, *generator,
         "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_INSTALL_PREFIX={prefix}",
         "-DBUILD_TESTS=OFF", "-DBUILD_PERF_TESTS=OFF", "-DBUILD_EXAMPLES=OFF",
         "-DBUILD_opencv_apps=OFF", "-DBUILD_JAVA=OFF", "-DBUILD_opencv_python3=OFF"])

    # Explicit job count: a bare --parallel means unlimited jobs with Makefiles, which can run out of memory.
    run(["cmake", "--build", build_dir, "--parallel", str(os.cpu_count() or 2)])
    run(["cmake", "--install", build_dir])


def install_opencv_linux(deps_dir: Path, version: str) -> Path:
    """Build and install OpenCV unless already there. Return the folder that holds OpenCVConfig.cmake."""
    prefix = deps_dir / f"opencv-{version}"

    if is_installed(prefix):
        print(f"OpenCV {version} already installed: {prefix}")
    else:
        print("Building OpenCV from source. Needs git, cmake, a C++ compiler, and the GTK3/FFmpeg dev packages.")
        work = deps_dir / "downloads" / f"opencv-{version}"
        shutil.rmtree(work, ignore_errors=True)    # Leftovers of an interrupted run.
        shutil.rmtree(prefix, ignore_errors=True)
        clone_opencv(version, work / "src")
        build_opencv(work / "src", work / "build", prefix)
        shutil.rmtree(work)
        mark_installed(prefix)

    return find_config(prefix, "OpenCVConfig.cmake")


# ----- Main -----

def configure_hint(system: str, deps_dir: Path, args: argparse.Namespace) -> str:
    """How to configure. cmake/dependencies.cmake finds the install folder by itself, unless it was moved."""
    hint = f"cmake --preset {CONFIGURE_PRESET[system]}"
    if deps_dir != default_deps_dir(system).resolve() and "TENGEN_DEPS_DIR" not in os.environ:
        hint += f' "-DTENGEN_DEPS_DIR={deps_dir.as_posix()}"'
    if args.qt_arch != QT_DEFAULT_ARCH[system]:
        hint += f" -DTENGEN_QT_ARCH={args.qt_arch}"
    # An existing build folder caches the paths of a previous Qt/OpenCV install.
    return hint + "\n  (add --fresh if the build folder was configured with other Qt or OpenCV versions before)"


def main() -> None:
    # 1) Check platform and parse arguments
    system = check_platform()
    args = parse_args(system)
    deps_dir = args.deps_dir.expanduser().resolve()

    # 2) Parse CMakeLists for versions
    versions = read_versions()
    print(f"Installing into {deps_dir}\n"
          f"  Qt     {versions.qt}\n"
          f"  OpenCV {versions.opencv}{' (skipped)' if args.skip_opencv else ''}\n")

    try:
        # 3) Install Qt
        qt_dir = install_qt(deps_dir, versions.qt, args.qt_arch)
        print(f"Qt:     {qt_dir}")

        # 4) Install OpenCV
        if not args.skip_opencv:
            install_opencv = install_opencv_windows if system == "Windows" else install_opencv_linux
            print(f"OpenCV: {install_opencv(deps_dir, versions.opencv)}")
    except urllib.error.URLError as error:
        sys.exit(f"error: download failed: {error}")
    except subprocess.CalledProcessError as error:
        sys.exit(f"error: command failed with exit code {error.returncode}, see the output above.")

    # 5) Print how to configure
    print(f"\nDone. Configure with:\n  {configure_hint(system, deps_dir, args)}")


if __name__ == "__main__":
    main()
