# Game Detection Library (libCamera)

## Assumptions
To get a simple algorithm working for 1) detecting the board in an image 2) detecting the grid lines 3) detecting the stones, we work under some assumptions which may be weakened in the future.
 1) The board position in the image is fixed for a whole game (no bumping the table or moving the camera during game)
 2) User tunes the camera position and environment

The first assumption allows us to only properly detect the board one, then use this homography for the whole game.
The second assumption allows for the algorithm to not yet be hardened against lighting changes, strong angles and other factors which would make board detection more difficult.

The current goal is to provide a basic image detection algorithm which allows for a game to be captured.
An application to tune the algorithm parameters is provided (CameraTuner).

## Procedure
Let:
 - $W$ denote image width, $H$ image height and $C$ number of channels.
 - $I \subset \mathbb{R}^{H\times W\times C}$ denote the space of digital image.
 - $I_B \subset I$ dnote the subset of images containing a visible Go board.
 - $B \subset \mathbb{R}^2$ denote the canonical board coordinate domain (a square region with known grid structure).
 - $B_0 \subset \mathbb{R}^2$ denote the coarse board domain estimate, obtained from rough contour detection.

Given an image $i \in I_B$, the goal is to estimate a homography
$$ H: \mathbb{R}^2 \to \mathbb{R}^2 $$
such that points on the board plane in the image are mapped to a canonical board coordinate system.

The final homography $H$ is constructed by two consequetive steps in the pipeline $H = H_1 \circ H_0$.
 - **BoardFinder (coarse localization)** A rough quadrilateral is detected in the input image yielding a coarse homography
 $$
    \begin{align*}
        H_0: I_B &\to B_0 \\ i &\mapsto i_0
    \end{align*}
 $$
 - **GridFinder (refinement)** Grid lines are detected in the coarse warp $i_0$ and used to refine the alignment, producing
 $$
    \begin{align*}
        H_1 : B_0 &\to B \\ i_0 &\mapsto i_B
    \end{align*}
 $$

Under the assumption that the board pose remains fixed over time, a single calibration image $i\in I_B$ determines $H$, which may then be reused for subsequent frames until re-calibration is requested.

Finally, **StoneFinder** operates in canonical board coordinates $B$ to detect per-intersection stone states.

Unfortunately, image detection is not closed-form exactly solvable so the given algorithm does not work on $I_B$ but rather on some $I_{B_0}\subset I_B$ which we aim to maximize with continuous improvements ;).
