#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <vector>

namespace tengen::labeler {

struct Pixel {
	int x;
	int y;
};

//! The labelled geometry of one photo, see tests/vision/resources/README.md.
struct Geometry {
	unsigned boardSize{0};
	std::array<Pixel, 4> boardCorners{};
	std::array<Pixel, 4> gridCorners{};
};

//! The photos in `folder` that have no geometry file yet, sorted by name.
std::vector<std::filesystem::path> unlabelledPhotos(const std::filesystem::path& folder);

//! Where the geometry of a photo goes: next to it, as `<photo>.json`.
std::filesystem::path geometryFile(const std::filesystem::path& photo);

//! None if the file is missing or not a geometry file.
std::optional<Geometry> readGeometry(const std::filesystem::path& file);

//! Writes in the layout of the existing geometry files. False if the file cannot be written.
bool writeGeometry(const std::filesystem::path& file, const Geometry& geometry);

} // namespace tengen::labeler
