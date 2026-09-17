# Graphical User Interface

The GUI is the Qt6 front end for Tengen. It draws the board, takes clicks and button presses, and shows whatever state it's given — and that's it. It has no idea what a legal move is, it doesn't own the board, and it can't tell you whose turn it is unless something else tells it first.

## Widgets only know how to draw

`BoardWidget`, `GameWidget`, and `ChatWidget` are plain Qt widgets. Give them a board and they'll paint it; click on them and they'll emit a plain Qt signal saying "someone clicked here" or "someone hit pass." They don't check whether a move is legal, they don't know what a capture is, and they never reach out to the network. The actual painting is split off even further into its own `BoardRenderer` class, so the widget itself only has to deal with Qt events.

Because the widgets carry no game logic, they're reusable outside the actual game app. The board viewer tool, for instance, reuses the very same `BoardWidget` just to display a board — there's no `Game` involved at all.

## Presenters translate, they don't decide

Between the widgets and the actual game sits a small layer of presenters (`MainWindowPresenter`, `GamePresenter`, `BoardPresenter`, `ChatPresenter`). Their whole job is translation, in both directions:

- When a widget emits a signal, its presenter turns it into a request — place a stone, pass, resign, send a chat message — and passes it on.
- When the game reports that something changed, the presenter reads the new state and pushes it into the widget.

A presenter never decides whether a move is legal and never touches the board directly; it just relays. Updates can also arrive from a different thread than the GUI's — the game loop, or the network — so presenters are also where those updates get handed back to the GUI thread before touching any widget.

## The real game state never lives in the GUI

None of this — widgets or presenters — ever stores the actual game state.
Obviously the game logic is not here but the GUI layer also does not track the board position for example.
It all lives behind one small `GameSession` interface that the GUI talks to, and what's actually behind that interface depends on how the game was started:

- For a local game, it's a session wrapping a core `Game` directly — the same `Game` described in [Core.md](Core.md).
- For a network game, it's a session that talks to a server and keeps its own copy of whatever the server has told it. Hosting works the same way under the hood — starting a "host" spins up a real server in the background and connects to it exactly like a remote one would.

Because the GUI only ever talks to that one interface, it genuinely doesn't know or care whether it's driving a local game, hosting one, or connected to someone else's. Switching between them never touches a single line of widget or presenter code.
