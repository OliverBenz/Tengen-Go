# Core Library

The core library plays Go. It has no idea what a window, a button, or a network connection is — its only job is enforcing the rules and tracking what's happening on the board.

Everything revolves around a single `Game` object, the only place holding board state and turn order, so there's never a question of which copy of the game state is right. Anything that talks to a `Game` is a consumer — it sends a request (place a stone, pass, resign, shut down) rather than acting directly, and the `Game` processes requests one at a time in the order they arrive. Right now there are two consumers, a local session and a network server, going through the exact same interface.

Before acting, the `Game` checks the request against the real rules: turn order, whether the point is legal, the suicide rule, and the ko rule. Illegal requests are simply ignored with no error or reply.

Accepted moves get reported two ways: **signals**, a cheap "something changed" ping for UI refresh, and **deltas**, the actual data (move number, type, player, captures, next turn) that consumers use to rebuild their own view instead of polling the `Game`. Both fire in order right after the request is processed. Ending a game is just another update — two passes or a resignation, and the `Game` reports itself finished like any other change.
