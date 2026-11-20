#include "vision/core/stoneClassifier.hpp"

#include "stoneNetModel.hpp"

#include <opencv2/dnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace tengen::vision::core {

// See resource/models/stoneNet.json.
static constexpr std::array<StoneState, 3> CLASSES{StoneState::Empty, StoneState::Black, StoneState::White}; //!< Logit order.
static constexpr double SCALE = 1.0 / 255.0;                                                                 //!< To [0, 1].

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

std::vector<Classification> StoneClassifier::classify(const std::vector<cv::Mat>& patches) {
	if (patches.empty()) {
		return {};
	}
	const bool fits = std::ranges::all_of(patches, [](const cv::Mat& patch) { return patch.rows == PATCH_SIZE && patch.cols == PATCH_SIZE && patch.type() == CV_8UC3; });
	if (!fits) {
		throw std::invalid_argument("Stone patches must be " + std::to_string(PATCH_SIZE) + "x" + std::to_string(PATCH_SIZE) + " 8-bit BGR images");
	}

	m_impl->net.setInput(cv::dnn::blobFromImages(patches, SCALE, cv::Size(), cv::Scalar(), /*swapRB=*/true)); // NCHW, RGB.
	const cv::Mat logits = m_impl->net.forward();
	if (logits.dims != 2 || logits.type() != CV_32F || logits.rows != static_cast<int>(patches.size()) || logits.cols != static_cast<int>(CLASSES.size())) {
		throw std::runtime_error("The stone model's output does not match stoneNet.json");
	}

	std::vector<Classification> result;
	result.reserve(patches.size());
	for (int i = 0; i < logits.rows; ++i) {
		const float* scores    = logits.ptr<float>(i);
		const std::size_t best = static_cast<std::size_t>(std::max_element(scores, scores + CLASSES.size()) - scores);
		float sum              = 0.f;
		for (std::size_t k = 0; k < CLASSES.size(); ++k) {
			sum += std::exp(scores[k] - scores[best]); // Shifted: no overflow.
		}
		result.push_back({.state = CLASSES[best], .confidence = 1.f / sum});
	}
	return result;
}

} // namespace tengen::vision::core
