#!/usr/bin/env python3
"""Install Qt and OpenCV on Windows. Linux uses the system packages instead.

Everything goes into a shared folder outside the repository, so deleting build/ never forces a reinstall.
Each version gets its own subfolder, so a version bump installs next to the old one.
Running the script again skips whatever is already installed. Delete a version's folder to reinstall it.

Qt:     Official prebuilt binaries, downloaded with aqtinstall (installed into a private venv).
OpenCV: Official prebuilt release.

Usage:  python scripts/win/1_setup_deps.py [--deps-dir DIR] [--qt-arch ARCH] [--skip-opencv]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ----- Dependency information -----

# The versions we test and ship. CMakeLists.txt only requires a minimum, so other installs work too.
QT_VERSION = "6.12.0"
OPENCV_VERSION = "5.0.0"

# SHA-256 of each OpenCV version's Windows package, copied from its release page.
OPENCV_SHA256 = {
    "5.0.0": "9c6c1fcea58acdf06edba13148b2246e00c2658143fa51e61ecd370db8c39f63",
}

# aqtinstall and every package it pulls in, pinned with their hashes. See the file on how to regenerate it.
AQT_REQUIREMENTS = Path(__file__).with_name("aqt-requirements.txt")

QT_MODULES = ["qtmultimedia"]  # Add-on modules. Widgets and Concurrent are part of the base package.
QT_DEFAULT_ARCH = "win64_msvc2022_64"

STAMP = ".tengen-installed"  # Written last, so an interrupted install is never mistaken for a finished one.


# ----- Setup -----

def check_platform() -> None:
    """Exit if this is not x64 Windows."""
    if platform.system() == "Linux":
        sys.exit("Linux uses the system packages, e.g.: sudo pacman -S qt6-base qt6-multimedia opencv")
    if platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64"):
        sys.exit(f"error: only x64 Windows is supported (this is {platform.system()} {platform.machine()}).")


def default_deps_dir() -> Path:
    """Shared install folder in the user's data folder."""
    return Path(os.environ["LOCALAPPDATA"]) / "tengen" / "deps"


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Install the Qt and OpenCV versions Tengen is tested with.")
    parser.add_argument("--deps-dir", type=Path, default=os.environ.get("TENGEN_DEPS_DIR") or default_deps_dir(),
                        help="Install folder. Default: TENGEN_DEPS_DIR, else the user's data folder.")
    parser.add_argument("--qt-arch", default=QT_DEFAULT_ARCH, help="aqt architecture name.")
    parser.add_argument("--skip-opencv", action="store_true",
                        help="Only install Qt. Configuring still needs OpenCV until vision can be turned off.")
    return parser.parse_args()


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


def open_url(url: str, headers: dict[str, str] | None = None):
    """Open a URL. GitHub rejects requests without a User-Agent."""
    request = urllib.request.Request(url, headers={"User-Agent": "tengen-setup", **(headers or {})})
    return urllib.request.urlopen(request)


def download(url: str, target: Path, sha256: str) -> None:
    """Download to a .part file first, so a cut-off download never looks complete.
    Only a file matching the sha256 ever becomes the target."""
    print(f"Downloading {url}", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    with open_url(url) as response, open(part, "wb") as file:
        shutil.copyfileobj(response, file)
    with open(part, "rb") as file:
        digest = hashlib.file_digest(file, "sha256")

    if digest.hexdigest() != sha256.lower():
        part.unlink()
        sys.exit(f"error: {url} does not match the pinned SHA-256.\n"
                 f"  expected {sha256.lower()}\n"
                 f"  got      {digest.hexdigest()}")
    part.replace(target)


# ----- Qt -----

def ensure_aqt(deps_dir: Path) -> Path:
    """Return the Python of a private venv with the pinned aqtinstall. Rebuilds the venv when the pins change."""
    venv = deps_dir / "tools" / "aqt-venv"
    python = venv / "Scripts" / "python.exe"
    requirements = AQT_REQUIREMENTS.read_text(encoding="utf-8")

    if not is_installed(venv, requirements):
        shutil.rmtree(venv, ignore_errors=True)
        run([sys.executable, "-m", "venv", venv])
        # Uses the venv's own pip: upgrading it would pull an unpinned package.
        run([python, "-m", "pip", "install", "--quiet", "--require-hashes", "-r", AQT_REQUIREMENTS])
        mark_installed(venv, requirements)
    return python


def download_qt(python: Path, version: str, arch: str, out_dir: Path) -> None:
    """Install Qt with aqt into <out_dir>/<version>/."""
    try:
        run([python, "-m", "aqt", "install-qt", "windows", "desktop", version, arch,
             "-m", *QT_MODULES, "--outputdir", out_dir])
    except subprocess.CalledProcessError:
        sys.exit(f"error: Qt install failed, see the aqt output above. If the architecture is wrong, list them with:\n"
                 f"  {python} -m aqt list-qt windows desktop --arch {version}\n"
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


# ----- OpenCV -----

def find_opencv_asset(version: str) -> tuple[str, str]:
    """Look up name and URL of the official Windows package, instead of guessing the name."""
    # CI runners share IPs and quickly hit the anonymous API rate limit, a token lifts it.
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with open_url(f"https://api.github.com/repos/opencv/opencv/releases/tags/{version}", headers) as response:
        assets = json.load(response)["assets"]

    asset = next((a for a in assets if a["name"].endswith("-windows.exe")), None)
    if asset is None:
        sys.exit(f"error: OpenCV release {version} has no Windows package.")
    return asset["name"], asset["browser_download_url"]


def extract_opencv(installer: Path, target: Path) -> None:
    """Extract the self-extracting 7z package into target."""
    shutil.rmtree(target, ignore_errors=True)  # Leftovers of an interrupted extraction.
    run([installer, f"-o{target}", "-y"])


def install_opencv(deps_dir: Path, version: str) -> Path:
    """Install OpenCV unless already there. Return the folder that holds the top-level OpenCVConfig.cmake."""
    target = deps_dir / f"opencv-{version}"

    if is_installed(target):
        print(f"OpenCV {version} already installed: {target}")
    else:
        name, url = find_opencv_asset(version)
        installer = deps_dir / "downloads" / name
        download(url, installer, OPENCV_SHA256[version])
        extract_opencv(installer, target)
        installer.unlink()
        mark_installed(target)

    return find_config(target, "OpenCVConfig.cmake")


# ----- Main -----

def configure_hint(deps_dir: Path, args: argparse.Namespace) -> str:
    """How to configure. lib/dependencies.cmake finds the install folder by itself, unless it was moved."""
    hint = "cmake --preset Win64"
    if deps_dir != default_deps_dir().resolve() and "TENGEN_DEPS_DIR" not in os.environ:
        hint += f' "-DTENGEN_DEPS_DIR={deps_dir.as_posix()}"'
    if args.qt_arch != QT_DEFAULT_ARCH:
        hint += f" -DTENGEN_QT_ARCH={args.qt_arch}"
    # An existing build folder caches the paths of a previous Qt/OpenCV install.
    return hint + "\n  (add --fresh if the build folder was configured with other Qt or OpenCV versions before)"


def main() -> None:
    # 1) Check platform and parse arguments
    check_platform()
    args = parse_args()
    if not args.skip_opencv and OPENCV_VERSION not in OPENCV_SHA256:
        sys.exit(f"error: no SHA-256 pinned for OpenCV {OPENCV_VERSION}. Add it to OPENCV_SHA256.")
    deps_dir = args.deps_dir.expanduser().resolve()
    print(f"Installing into {deps_dir}\n"
          f"  Qt     {QT_VERSION}\n"
          f"  OpenCV {OPENCV_VERSION}{' (skipped)' if args.skip_opencv else ''}\n")

    try:
        # 2) Install Qt
        qt_dir = install_qt(deps_dir, QT_VERSION, args.qt_arch)
        print(f"Qt:     {qt_dir}")

        # 3) Install OpenCV
        if not args.skip_opencv:
            print(f"OpenCV: {install_opencv(deps_dir, OPENCV_VERSION)}")
    except urllib.error.URLError as error:
        sys.exit(f"error: download failed: {error}")
    except subprocess.CalledProcessError as error:
        sys.exit(f"error: command failed with exit code {error.returncode}, see the output above.")

    # 4) Print how to configure
    print(f"\nDone. Configure with:\n  {configure_hint(deps_dir, args)}")


if __name__ == "__main__":
    main()
