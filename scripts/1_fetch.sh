#!/bin/bash

# Images (GUI resources, vision test photos, training data) are in Git LFS.
if ! git lfs version >/dev/null 2>&1; then
	echo "git-lfs is not installed, install it first (e.g. pacman -S git-lfs)." >&2
	exit 1
fi

git submodule update --init --recursive
git lfs install --local
git lfs pull
