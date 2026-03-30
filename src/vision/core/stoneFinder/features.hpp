#pragma once

namespace tengen::vision::core {

struct Features {
	float deltaL{0.0f};
	float chromaSq{0.0f};
	float darkFrac{0.0f};
	float brightFrac{0.0f};
	bool valid{false};
};

} // namespace tengen::vision::core
