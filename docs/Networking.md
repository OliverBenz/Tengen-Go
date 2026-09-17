# Networking

There's still only one `Game` in a networked match — it just happens to live on someone else's machine, and every request has to cross a wire to reach it. Nothing about the rules changes: a consumer sends a request, the `Game` decides, and updates flow back out, exactly like the relationship described in [Core.md](Core.md).

## The server holds the only real Game

Only one place in a networked game actually runs a `Game`: the server. It owns the game loop, and it's the only thing anyone trusts. Clients never get their own copy of the rules — they just keep a local record of whatever the server has told them, so they have something to draw on screen while they wait for the next update.

Under the hood this is split into two pieces: a plain transport layer that just moves bytes over TCP and doesn't know Go exists, and a layer on top that turns those bytes into typed messages — place a stone, pass, resign, send a chat line — encoded as small JSON payloads. The split means the transport could be reused for something that has nothing to do with Go, and nobody working on game messages has to think about sockets.

## Clients propose, same as always

A client can't just place a stone — it sends the server a request, the same way any consumer asks the `Game` for something locally. The server hands that request straight to its `Game`, and the `Game` decides, exactly like it always does. If the request is illegal, nothing comes back — no rejection message, just silence, because that's how the `Game` behaves everywhere else too.

If the move is accepted, the `Game` produces its usual update, and the server packages that up and sends it to every connected client. That's the only way clients learn what happened — nobody predicts a move locally and hopes it matches; everybody just waits to be told.

## Seats decide what you're allowed to do

The first two people to connect to a game become Black and White. Everyone who connects after that is an Observer.

Observers see exactly what the players see — every move, every chat message — they just can't send anything back. No moves, no chat, nothing. They can watch a game unfold in real time without being able to touch it.

## Hosting is still just a server

Starting a hosted game doesn't run a different code path — it starts a real server in the background and connects to it exactly the way a remote player would, as covered in [GUI.md](GUI.md). There's no special "local host" mode hiding in the network code; hosting is just being the first client of your own server.
