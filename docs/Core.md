# Core Library

The core library is the part of Tengen that actually knows how to play Go. It has no idea what a window, a button, or a network connection is — its only job is enforcing the rules and keeping track of what's happening on the board.

## The Game is the single source of truth

Everything revolves around one class: the `Game`. It holds the board, keeps track of whose turn it is, and remembers enough history to enforce the rules properly. Nothing else in the codebase is allowed to change that state directly — if something needs to happen in a game, it has to go through the `Game`. There's no back door.

This is a deliberate choice. Because there's exactly one place where game state lives and changes, there's never a question of which copy is "right." Everything else — the local UI, the network server, and whatever we build next — just reacts to what the `Game` tells it.

## Consumers propose, the Game decides

We call anything that talks to a `Game` a consumer. A consumer doesn't reach in and place a stone itself — it sends the `Game` a request: place a stone here, pass, resign, or shut down entirely. The `Game` works through these requests in the order they arrive, one at a time, so there's never any doubt about what happened first.

Right now we have two consumers: a local session for playing on one machine, and a network server that relays moves between players connected remotely. Both talk to the `Game` in exactly the same way. The `Game` doesn't know or care which one is asking, and that's really the point — it keeps the rules in one place instead of spreading them across the UI and the network code.

## Every move is checked against the real rules

Before the `Game` acts on a request, it checks it against the actual rules of Go: is it this player's turn, is the point on the board and free, and does the move leave the stone — or the group it joins — without any liberties (empty neighboring points), unless it captures enemy stones in the process, which frees things up again. It also checks whether the resulting board position has shown up before anywhere in the game — this is what's usually called the superko rule, and it's what keeps ko fights and longer repeating cycles from looping forever.

If a request fails any of these checks, nothing happens. The `Game` just ignores it — no error, no special reply. A consumer that sends an illegal move simply notices that nothing changed.

## Signalling changes

Once a move is accepted, the `Game` tells the rest of the app about it in two ways:

- **Signals** are a quick heads-up that something changed — the board, the current player, or the overall game state. They're cheap and simple, meant for things like refreshing a UI element without needing the full picture.
- **Deltas** carry the real information: the move number, what kind of move it was, who made it, what got captured, and whose turn is next. Consumers use deltas to rebuild their own view of the game, so they never need to ask the `Game` what its internal state looks like.

Both go out the moment the `Game` finishes processing the request that caused them, in the same order they actually happened.
Consumers can track the current game state themselves via the `Deltas` they get from the game.

## The game ends the same way it plays

There's no special mechanism for ending a game — it's just another update. Two passes in a row, or a resignation, and the `Game` marks itself finished and reports that like any other change. Consumers find out the game is over the same way they find out a stone was placed: the `Game` tells them.
