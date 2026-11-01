# Releasing

Pushing a version tag runs [`release.yml`](../.github/workflows/release.yml), which builds the Linux and Windows packages and publishes them as a GitHub release.

## Steps

1. Set the version in `project(... VERSION x.y.z)` of [`CMakeLists.txt`](../CMakeLists.txt) and merge to `master`.
2. Tag the latest `master` and push the tag:

```
git switch master && git pull
git tag v0.5.0-alpha
git push origin v0.5.0-alpha
```

The tag must be `v` + the CMake version, optionally followed by a suffix. A suffix (`-alpha`, `-rc1`, ...) marks the release as a pre-release.

## What the pipeline does

1. Checks that the tag matches the CMake version.
2. Runs the checks of [`cmake.yml`](../.github/workflows/cmake.yml) and, alongside, builds and installs `Linux64-Release` (Arch container) and `Win64` Release (`windows-2022`).
3. Once all of them pass, publishes `tengen-<version>-linux-x64.tar.gz`, `tengen-<version>-windows-x64.zip` and `SHA256SUMS.txt`, with notes generated from the commits.

The Linux package uses the system Qt and OpenCV 5. The Windows package brings everything it needs.

A manual run (Actions → Release → Run workflow) does step 2 and leaves the packages as workflow artifacts, without publishing.

## Failed release

If the release was already published, delete it on GitHub first.
Then fix the problem and move the tag to the fixed commit:

```
git push --delete origin v0.5.0-alpha
git tag -f v0.5.0-alpha
git push origin v0.5.0-alpha
```
