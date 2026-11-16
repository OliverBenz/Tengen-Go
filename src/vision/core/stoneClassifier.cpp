#include "vision/core/stoneClassifier.hpp"

#include <opencv2/dnn.hpp>

#include <stdexcept>
#include <string>

namespace tengen::vision::core {

struct StoneClassifier::Impl {
	cv::dnn::Net net;
};

StoneClassifier::StoneClassifier(const std::filesystem::path& modelPath)
    : m_impl{std::make_unique<Impl>()} {
	try {
		m_impl->net = cv::dnn::readNetFromONNX(modelPath.string());
	} catch (const cv::Exception& e) {
		throw std::runtime_error("Cannot load stone classifier " + modelPath.string() + ": " + e.what());
	}
	if (m_impl->net.empty()) {
		throw std::runtime_error("Cannot load stone classifier " + modelPath.string() + ": empty network");
	}
}

StoneClassifier::~StoneClassifier() = default;

} // namespace tengen::vision::core
