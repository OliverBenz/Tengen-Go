#include "geometryFile.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace tengen::labeler {

static bool isPhoto(const std::filesystem::path& file) {
	std::string extension = file.extension().string();
	std::ranges::transform(extension, extension.begin(), [](const unsigned char c) { return std::tolower(c); });
	return extension == ".png" || extension == ".jpg" || extension == ".jpeg";
}

static std::string toJson(const std::array<Pixel, 4>& corners) {
	const auto& [a, b, c, d] = corners;
	return std::format("[\n    [{}, {}],\n    [{}, {}],\n    [{}, {}],\n    [{}, {}]\n  ]", a.x, a.y, b.x, b.y, c.x, c.y, d.x, d.y);
}

static std::array<Pixel, 4> toCorners(const nlohmann::json& points) {
	std::array<Pixel, 4> corners{};
	for (std::size_t i = 0; i < corners.size(); ++i) {
		const nlohmann::json& point = points.at(i);
		corners[i]                  = {static_cast<int>(std::lround(point.at(0).get<double>())), static_cast<int>(std::lround(point.at(1).get<double>()))};
	}
	return corners;
}

std::vector<std::filesystem::path> unlabelledPhotos(const std::filesystem::path& folder) {
	std::vector<std::filesystem::path> photos;
	for (const auto& entry: std::filesystem::directory_iterator(folder)) {
		if (entry.is_regular_file() && isPhoto(entry.path()) && !std::filesystem::exists(geometryFile(entry.path()))) {
			photos.push_back(entry.path());
		}
	}
	std::ranges::sort(photos);
	return photos;
}

std::filesystem::path geometryFile(const std::filesystem::path& photo) {
	return std::filesystem::path(photo).replace_extension(".json");
}

std::optional<Geometry> readGeometry(const std::filesystem::path& file) {
	try {
		const nlohmann::json json = nlohmann::json::parse(std::ifstream(file));
		const Geometry geometry{
		        .boardSize    = json.at("boardSize").get<unsigned>(),
		        .boardCorners = toCorners(json.at("boardCorners")),
		        .gridCorners  = toCorners(json.at("gridCorners")),
		};
		if (geometry.boardSize != 9u && geometry.boardSize != 13u && geometry.boardSize != 19u) {
			return std::nullopt;
		}
		return geometry;
	} catch (const nlohmann::json::exception&) {
		return std::nullopt;
	}
}

bool writeGeometry(const std::filesystem::path& file, const Geometry& geometry) {
	std::ofstream out(file);
	out << std::format("{{\n  \"boardSize\": {},\n  \"boardCorners\": {},\n  \"gridCorners\": {}\n}}\n",
	                   geometry.boardSize, toJson(geometry.boardCorners), toJson(geometry.gridCorners));
	out.close();
	return !out.fail();
}

} // namespace tengen::labeler
