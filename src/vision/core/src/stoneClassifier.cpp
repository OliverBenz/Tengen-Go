#include "vision/core/stoneClassifier.hpp"

#include "stoneNetModel.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tengen::vision::core {

// See resource/models/stoneNet.json.
static constexpr std::array<StoneState, 3> CLASSES{StoneState::Empty, StoneState::Black, StoneState::White}; //!< Logit order.
static constexpr double SCALE = 1.0 / 255.0;                                                                 //!< To [0, 1].

static constexpr std::size_t BATCH_SIZE = 64; //!< Patches per network run: bounds the memory used at once.

static cv::dnn::Net loadModel() {
	const std::span<const unsigned char> model = stoneNetModel();
	try {
		cv::dnn::Net net = cv::dnn::readNetFromONNX(std::vector<uchar>(model.begin(), model.end()));
		if (!net.empty()) {
			return net;
		}
	} catch (const cv::Exception& e) {
		throw std::runtime_error(std::string("Cannot load the stone model: ") + e.what());
	}
	throw std::runtime_error("Cannot load the stone model: empty network");
}

struct StoneClassifier::Impl {
	cv::dnn::Net net;
};

StoneClassifier::StoneClassifier()
    : m_impl(std::make_unique<Impl>(Impl{.net = loadModel()})) {
	// Fail at startup if the model does not fit.
	classify({cv::Mat(PATCH_SIZE, PATCH_SIZE, CV_8UC3, cv::Scalar::all(0))});
}

StoneClassifier::~StoneClassifier() = default;

//! Runs one batch through the network, appending its classifications.
static void classifyBatch(cv::dnn::Net& net, const std::vector<cv::Mat>& batch, std::vector<Classification>& result) {
	net.setInput(cv::dnn::blobFromImages(batch, SCALE, cv::Size(), cv::Scalar(), /*swapRB=*/true)); // NCHW, RGB.
	const cv::Mat logits = net.forward();
	if (logits.dims != 2 || logits.type() != CV_32F || logits.rows != static_cast<int>(batch.size()) || logits.cols != static_cast<int>(CLASSES.size())) {
		throw std::runtime_error("The stone model's output does not match stoneNet.json");
	}

	for (int i = 0; i < logits.rows; ++i) {
		const float* scores    = logits.ptr<float>(i);
		const std::size_t best = static_cast<std::size_t>(std::max_element(scores, scores + CLASSES.size()) - scores);
		float sum              = 0.f;
		for (std::size_t k = 0; k < CLASSES.size(); ++k) {
			sum += std::exp(scores[k] - scores[best]); // Shifted: no overflow.
		}
		result.push_back({.state = CLASSES[best], .confidence = 1.f / sum});
	}
}

std::vector<Classification> StoneClassifier::classify(const std::vector<cv::Mat>& patches) {
	const bool fits = std::ranges::all_of(patches, [](const cv::Mat& patch) { return patch.rows == PATCH_SIZE && patch.cols == PATCH_SIZE && patch.type() == CV_8UC3; });
	if (!fits) {
		throw std::invalid_argument("Stone patches must be " + std::to_string(PATCH_SIZE) + "x" + std::to_string(PATCH_SIZE) + " 8-bit BGR images");
	}

	std::vector<Classification> result;
	result.reserve(patches.size());
	const std::span<const cv::Mat> all(patches);
	for (std::size_t first = 0; first < all.size(); first += BATCH_SIZE) {
		const std::span<const cv::Mat> batch = all.subspan(first, std::min(BATCH_SIZE, all.size() - first));
		classifyBatch(m_impl->net, {batch.begin(), batch.end()}, result); // Copies the Mat headers only.
	}
	return result;
}

} // namespace tengen::vision::core
