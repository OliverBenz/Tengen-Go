#pragma once

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace tengen::vision::core {

//! Stone state at a single grid intersection.
enum class StoneState {
	Empty,
	Black,
	White
};

//! What the classifier made of one patch.
struct Classification {
	StoneState state{StoneState::Empty};
	float confidence{0.f}; //!< Softmax probability of state, in [0, 1].
};

//! The built-in stoneNet CNN (resource/models/stoneNet.json). Not thread-safe: one per thread.
class StoneClassifier {
public:
	static constexpr int PATCH_SIZE = 32; //!< [px] Patch side and intersection spacing.

	StoneClassifier(); //!< \throws std::runtime_error if the model does not load.
	~StoneClassifier();

	StoneClassifier(const StoneClassifier&)            = delete; // Copies would share the network.
	StoneClassifier& operator=(const StoneClassifier&) = delete;

	//! One result per patch: 8-bit BGR, PATCH_SIZE wide, centred on an intersection.
	std::vector<Classification> classify(const std::vector<cv::Mat>& patches);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl; //!< Hides dnn.
};

} // namespace tengen::vision::core
