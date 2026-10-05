# Core Library

The core library plays Go. It has no idea what a window, a button, or a network connection is — its only job is enforcing the rules and tracking what's happening on the board.

Everything revolves around a single `Game` object, the only place holding board state and turn order, so there's never a question of which copy of the game state is right. Anything that talks to a `Game` is a consumer — it sends a request (start, place a stone, pass, resign, shut down) rather than acting directly, and the `Game` processes requests one at a time in the order they arrive. Right now there are three consumers, a local session, a bot session and a network server, going through the exact same interface.

A `Game` is set up with its config (board size and rules) but takes no moves until it is started. That lets a consumer create the game early and start it once everyone is ready: the bot engine is up, or both players are connected.

Before acting, the `Game` checks the request against the real rules: turn order, whether the point is legal, the suicide rule, and the ko rule. Illegal requests are simply ignored with no error or reply.

Changes get reported two ways: **signals**, a cheap "something changed" ping for UI refresh, and **state updates**, the actual data that consumers use to rebuild their own view instead of polling the `Game`. Both fire in order right after the request is processed. The state updates tell the whole game: a **start** with the config, one **delta** per accepted move (move number, type, player, captures, next turn), and an **end** with the result (the winner, if there is one, and why the game ended) right after the move that ended it — two passes or a resignation. A consumer has to listen from before the start, or it misses the config.
