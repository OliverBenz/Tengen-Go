#pragma once

#include <filesystem>
#include <memory>

namespace tengen::vision::core {

//! CNN classifying intersections as empty, black or white (models/stoneNet.json).
class StoneClassifier {
public:
	//! \throws std::runtime_error if modelPath is no readable ONNX model.
	explicit StoneClassifier(const std::filesystem::path& modelPath);
	~StoneClassifier();

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl; //!< Keeps dnn headers private.
};

} // namespace tengen::vision::core
