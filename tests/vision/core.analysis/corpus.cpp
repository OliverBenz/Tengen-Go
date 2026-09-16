#include "corpus.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <nlohmann/json.hpp>

namespace tengen::vision::analysis {
namespace {

bool isImage(const std::filesystem::path& path) {
	const auto extension = path.extension().string();
	return extension == ".png" || extension == ".jpeg" || extension == ".jpg";
}

//! Order "move_2" before "move_10" by comparing digit runs as numbers.
bool naturalLess(const std::string& lhs, const std::string& rhs) {
	std::size_t i = 0u;
	std::size_t j = 0u;
	while (i < lhs.size() && j < rhs.size()) {
		const bool bothDigits = std::isdigit(static_cast<unsigned char>(lhs[i])) && std::isdigit(static_cast<unsigned char>(rhs[j]));
		if (bothDigits) {
			std::size_t lhsEnd = i;
			std::size_t rhsEnd = j;
			while (lhsEnd < lhs.size() && std::isdigit(static_cast<unsigned char>(lhs[lhsEnd]))) {
				++lhsEnd;
			}
			while (rhsEnd < rhs.size() && std::isdigit(static_cast<unsigned char>(rhs[rhsEnd]))) {
				++rhsEnd;
			}

			const unsigned long lhsValue = std::stoul(lhs.substr(i, lhsEnd - i));
			const unsigned long rhsValue = std::stoul(rhs.substr(j, rhsEnd - j));
			if (lhsValue != rhsValue) {
				return lhsValue < rhsValue;
			}
			i = lhsEnd;
			j = rhsEnd;
		} else if (lhs[i] != rhs[j]) {
			return lhs[i] < rhs[j];
		} else {
			++i;
			++j;
		}
	}
	return (lhs.size() - i) < (rhs.size() - j);
}

Corners parseCorners(const nlohmann::json& array) {
	Corners corners{};
	for (std::size_t i = 0u; i < corners.size(); ++i) {
		corners[i] = {array.at(i).at(0).get<float>(), array.at(i).at(1).get<float>()};
	}
	return corners;
}

} // namespace

std::vector<ImageSet> discoverImageSets(const std::filesystem::path& root) {
	std::vector<ImageSet> sets;
	if (!std::filesystem::is_directory(root)) {
		return sets;
	}

	for (const auto& entry: std::filesystem::recursive_directory_iterator(root)) {
		if (!entry.is_directory()) {
			continue;
		}

		ImageSet set{};
		set.name = std::filesystem::relative(entry.path(), root).generic_string();
		for (const auto& file: std::filesystem::directory_iterator(entry.path())) {
			if (file.is_regular_file() && isImage(file.path())) {
				set.images.push_back(file.path());
			}
		}
		if (set.images.empty()) {
			continue;
		}

		std::sort(set.images.begin(), set.images.end(), [](const std::filesystem::path& lhs, const std::filesystem::path& rhs) {
			return naturalLess(lhs.filename().string(), rhs.filename().string());
		});
		set.labelled = loadGroundTruth(set.images.front()).has_value();
		sets.push_back(std::move(set));
	}

	std::sort(sets.begin(), sets.end(), [](const ImageSet& lhs, const ImageSet& rhs) { return lhs.name < rhs.name; });
	return sets;
}

std::optional<GroundTruth> loadGroundTruth(const std::filesystem::path& imagePath) {
	std::filesystem::path jsonPath = imagePath;
	jsonPath.replace_extension(".json");

	std::ifstream stream(jsonPath);
	if (!stream) {
		return std::nullopt;
	}

	const nlohmann::json json = nlohmann::json::parse(stream, nullptr, false);
	if (json.is_discarded() || !json.contains("boardCorners") || !json.contains("gridCorners")) {
		return std::nullopt;
	}

	return GroundTruth{json.value("boardSize", 0u), parseCorners(json.at("boardCorners")), parseCorners(json.at("gridCorners"))};
}

} // namespace tengen::vision::analysis
