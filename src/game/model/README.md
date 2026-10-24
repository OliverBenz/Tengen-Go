# Game Model
This library models the data of a real game of go: the physical objects and the abstract concepts a game needs.
Picture a room where two people sit down to play.

Physically, a game needs players, a board with coordinates, and a clock.
The players may also talk across the board, and what they say comes in a certain order.
On the abstract side, a game follows rules, every move changes the state of the game, and the game ends with a result.

What I deliberately left out for now is the room around the game -- the spectators, for instance, who may chat as well.
Their chat should not be audible to the players (spectators whisper).
If we ever add chat between spectators, we should decide whether this library describes only the players' inner circle or the whole room.
At the time of writing, I think a simple subdirectory `model/room` in the public include directory would be enough.

Keep these ideas in mind when you add something to the model library.
