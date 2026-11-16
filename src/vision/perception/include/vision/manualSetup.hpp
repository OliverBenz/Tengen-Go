#pragma once

#include "model/board.hpp"

#include <cstddef>
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
	Image warped;    //!< The frame warped onto the applied corners, the first one picked at the top left.
	Image rectified; //!< The rectified board with its grid points and the stones of board marked. Empty as long as no grid is found.

	//! The stones on the board: one shows once found for a few frames in a row. Size 0 as long as no grid is found.
	//! \note In the game's orientation, if the first corner picked is the one the game shows at the top left: that is Coord{0, 0}, as in rectified.
	Board board;
};

//! Set the board up by hand: the user picks the 4 board corners on the camera image instead of the board finder.
//! Once applied, the grid is found once and the stones on it are watched frame by frame.
class ManualSetup {
public:
	static constexpr std::size_t CORNER_COUNT = 4u;

	struct Callbacks {
		std::function<void(const Image&)> onFrame;            //!< A camera frame, until the corners are applied.
		std::function<void(const BoardSnapshot&)> onAnalysed; //!< A frame through the applied corners.
		std::function<void()> onCameraLost;                   //!< The camera could not be opened or stopped delivering.
	};

	//! Starts the camera right away. Callbacks run on its thread.
	//! \throws std::runtime_error if modelPath is no readable stone model.
	ManualSetup(int cameraIndex, const std::filesystem::path& modelPath, Callbacks callbacks);
	~ManualSetup(); //!< No callback once this returns.

	//! Add a board corner, in frame coordinates. False once all 4 are picked.
	//! The first one tells from which side the camera looks: it is where the game has its top left, Coord{0, 0}. The others may come in any order.
	bool pick(float x, float y);
	std::size_t picked() const; //!< Corners picked so far.
	bool apply();               //!< Fix H_0 to the picked corners. The first grid found in them is kept until redo(). False until all 4 are picked.
	void redo();                //!< Drop the corners and their grid, and go back to picking.
	bool isApplied() const;

private:
	struct State;
	std::unique_ptr<State> m_state; //!< Hides the camera and the pipeline.
};

} // namespace tengen::vision
