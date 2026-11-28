#include "vision/core/stoneClassifier.hpp"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace tengen::vision::core {
namespace gtest {

static constexpr int SIZE = StoneClassifier::PATCH_SIZE;

TEST(StoneClassifierUnit, Batching_KeepsResults) {
	cv::RNG rng(1);
	std::vector<cv::Mat> patches(150); // Spans several batches, the last one partial.
	for (cv::Mat& patch: patches) {
		patch.create(SIZE, SIZE, CV_8UC3);
		rng.fill(patch, cv::RNG::UNIFORM, 0, 256);
	}

	StoneClassifier classifier;
	const std::vector<Classification> batched = classifier.classify(patches);
	ASSERT_EQ(batched.size(), patches.size());
	for (std::size_t i = 0; i < patches.size(); ++i) {
		const Classification alone = classifier.classify({patches[i]}).front();
		EXPECT_EQ(batched[i].state, alone.state) << "patch " << i;
		EXPECT_NEAR(batched[i].confidence, alone.confidence, 1e-4f) << "patch " << i;
	}
}

TEST(StoneClassifierUnit, NoPatches_NoResults) {
	StoneClassifier classifier;
	EXPECT_TRUE(classifier.classify({}).empty());
}

TEST(StoneClassifierUnit, WrongPatchSize_Throws) {
	StoneClassifier classifier;
	EXPECT_THROW(classifier.classify({cv::Mat(SIZE + 1, SIZE, CV_8UC3, cv::Scalar::all(0))}), std::invalid_argument);
}

} // namespace gtest
} // namespace tengen::vision::core
