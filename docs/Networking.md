# Networking

There's still only one `Game` in a networked match, it just lives on someone else's machine and every request has to cross a wire to reach it — same relationship as in [Core.md](Core.md).

Only the server runs a `Game`; it owns the game loop and is the only thing anyone trusts. Clients keep a local record of whatever the server told them, just enough to draw the board. This is split into a plain TCP transport that knows nothing about Go, and a layer on top that turns bytes into typed JSON messages (place, pass, resign, chat) — so the transport is reusable and nobody working on game messages has to think about sockets.

A client sends a request like any consumer; the server hands it to its `Game`, which decides exactly like it always does, and illegal requests get the same silence as everywhere else. Accepted moves get packaged by the server and broadcast to every connected client — nobody predicts a move locally, everyone waits to be told.

The first two people to connect become Black and White; everyone after that is an Observer, seeing every move and chat message but unable to send anything back.

Hosting doesn't take a different code path: it starts a real server in the background and connects to it exactly like a remote player would (see [GUI.md](GUI.md)) — there's no separate "local host" mode, hosting just means being the first client of your own server.
