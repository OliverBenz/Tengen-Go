# Go Camera
This is the interface library connecting the `vision::core` OpenCV board detection code with the Go game we work on.
This means:
 - The `vision::core` returns a vector of stone values. Here, we translate this to a Board state.
 - The physical board - detected by the libCamera - is invariant under $D_4$ symmetry transformations. Once a stone (or two stones if the first is in center) is placed, the symmetry is broken and we can map to our expected board coordinates.
 - Here, we keep OpenCV a private dependency and do not propagate further.

We do not merge this library with the `vision::core` to keep `vision::core` useful to other people who do not want our data structures imposed.
This also allows us to frequently update/improve or replace the `vision::core` without affecting higher-level projects.

Summary:
 - The `vision::core` provides a general image detection algorithm of a Go board.
 - This project handles mapping of degenerate image data to `model::Board` states


## Gauge Symmetry
The camera may be places at any side of the Go board and the camera may flip the image.
Our go game system maps each intersection of the board to a fixed coordinate while the physical board does not care about the orientation.
This corresponds to a $D_4$ symmetry of the board.

In this `vision::perception` algorithm, we take the detected board and and stones from the `vision::core` algorithm and break this gauge symmetry.
This is done in an initial setup stage where the user places a single black stone onto a non-degenerate coordinate $c_s$ (not in the center).
The user then assigns a coordinate to this stone (e.g. Place at $c_s = A4$).
Our `vision::core` detects the same stone at some coordinate $c$.

We then generate the gauge orbit
$$\mathcal{O}_c^{D_4} = \{g \rhd c \; | \; g\in D_4\} $$
of this `vision::core` coordiante $c$ under $D_4$ and find the group element $g$ which maps $c_s = g \rhd c$.
This $g$ is the transformation which maps the image coordinate to the system coordinate.
If $c_s \notin \mathcal{O}_c^{D_4}$, then the algorithm fails as no symmetry transformation can map the detected stone onto the user coordinate.


## Design Choices
## Callback functions vs Event Handler.
Event Handler (like in the `game::core`): Signals updates to observers, then observers query new data from the signaling entity.
I prefer an event handler pattern for larger numbers of signals and if the signaling entity is a source of truth.

There are few events that occur in the image detection loop:
 - New stone detected
 - Board lost
 - Grid lost
and vision signals are flakey.
They should be seen as "hey, i might have found something" and not as a "there is a definite change of state".

Thus we are working with callback functions here.
