// Reports what BoardFinder and GridFinder produce for every test image.
// Run it before and after a change to either stage, then diff the two outputs.
#include "corpus.hpp"
#include "measure.hpp"
#include "report.hpp"

#include <algorithm>
#include <iostream>

int main(int argc, char** argv) {
	using namespace tengen::vision::analysis;

	const std::filesystem::path root = PATH_TEST_IMG;
	const std::vector<std::string> requestedSets(argv + 1, argv + argc);

	const std::vector<ImageSet> sets = discoverImageSets(root);
	if (sets.empty()) {
		std::cerr << "No image sets found under " << root << '\n';
		return 1;
	}

	std::size_t reported = 0u;
	for (const auto& set: sets) {
		if (!requestedSets.empty() && std::find(requestedSets.begin(), requestedSets.end(), set.name) == requestedSets.end()) {
			continue;
		}

		std::vector<Measurement> measurements;
		measurements.reserve(set.images.size());
		for (const auto& image: set.images) {
			measurements.push_back(measureImage(image, loadGroundTruth(image)));
		}

		if (set.labelled) {
			printLabelledReport(set, measurements);
		} else {
			printSeriesReport(set, measurements);
		}
		++reported;
	}

	if (reported == 0u) {
		std::cerr << "No set matched. Available sets:\n";
		for (const auto& set: sets) {
			std::cerr << "  " << set.name << '\n';
		}
		return 1;
	}

	return 0;
}
