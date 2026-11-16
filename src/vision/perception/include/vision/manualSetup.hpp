#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace tengen::vision {

//! An 8 bit RGB image, row after row.
struct Image {
	int width{0};
	int height{0};
	std::vector<std::uint8_t> rgb{};
};

//! What the pipeline made of one frame.
struct BoardSnapshot {
	Image warped;       //!< The frame warped onto the applied corners.
	Image rectified;    //!< The rectified board with its grid points and stones marked. A stone shows once found for a few frames in a row.
	unsigned boardSize; //!< Lines per side of the grid found. 0 if none was found, and rectified stays empty.
};

//! Set the board up by hand: the user picks the 4 board corners on the camera image instead of the board finder.
class ManualSetup {
public:
	struct Callbacks {
		std::function<void(const Image&)> onFrame;            //!< A camera frame, until the corners are applied.
		std::function<void(const BoardSnapshot&)> onAnalysed; //!< A frame through the applied corners.
		std::function<void()> onCameraLost;                   //!< The camera could not be opened or stopped delivering.
	};

	//! Starts the camera right away. Callbacks run on its thread.
	//! \throws std::runtime_error if modelPath is no readable stone model.
	ManualSetup(int cameraIndex, const std::filesystem::path& modelPath, Callbacks callbacks);
	~ManualSetup(); //!< No callback once this returns.

	bool pick(float x, float y); //!< Add a board corner, in frame coordinates. False once all 4 are picked.
	bool apply();                //!< Fix H_0 to the picked corners. False until all 4 are picked.
	void redo();                 //!< Drop the corners and go back to picking.
	bool isApplied() const;

private:
	struct State;
	std::unique_ptr<State> m_state; //!< Hides the camera and the pipeline.
};

} // namespace tengen::vision
