#pragma once

#include "model/coordinate.hpp"
#include "vision/source.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace tengen::vision {

struct StonePos {
	Coord c;
	float confidence;
};

/*! Manages the board and stone detection during the game.
 *  Process: The user
 *   - Calls Vision.setup(gaugeCoord) once. Homography and grid is stored and assumed static for the remaining game.
 *     The board has a natural D_4 symmetry. In the setup process, we have a single black stone placed on the board (not in the center!)
 *     to break the symmetry / to align the image rotation with the coordinate system.
 *   - Connects callback functions to be notified on different events.
 *   - Calls run() to start the board detection loop.
 */
class Vision {
public:
	struct Callbacks {
		std::function<void(StonePos)> onStonePlaced; //!< Callback when the image.
		std::function<void()> onBoardLost;           //!< Callback when the board or grid cannot be found anymore.
	};

public:
	explicit Vision(Source source = Source::Video);
	~Vision();

	//! Detect the board and grid once before the game starts.
	//! \param [in] gaugeStone Coordinate of a single black stone placed on the board. Used to fix the image orientation/coordinate mapping.
	//! \returns    True if the symmetry transformation $g$ could be found.
	bool setup(Coord gaugeCoord);
	void setSetupImage(std::filesystem::path setupImagePath); //!< Setup image path used when source is Source::Camera.

	void connect(Callbacks callbacks); //!< Connect callback functions.
	void disconnect();                 //!< Disconnect the callback functions.

	//! Start the stone detection loop. Calls back to onStonePlaced when a new stone is detected.
	//! \note Only runs if source is a live feed.
	void run();
	void stop();

private:
	struct State;
	std::unique_ptr<State> m_state; //!< Hides core and OpenCV.
};

} // namespace tengen::vision
