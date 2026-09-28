# Graphical User Interface

The GUI is the Qt6 front end for Tengen, built around a strict separation between the GUI layer and the business logic (BL) behind it. That separation is enforced with a presenter pattern: widgets only draw data and forward user interaction, presenters are the only thing that translates between the two sides.

`BoardWidget`, `GameWidget`, and `ChatWidget` are plain Qt widgets: give them a board and they paint it, click on them and they emit a plain Qt signal ("clicked here", "hit pass"). No legality checks, no capture logic, no network. Painting itself is split out into `BoardRenderer`, so the widgets carry no BL logic and are reusable outside the game app — the board viewer tool reuses `BoardWidget` with no `Game` involved at all.

Between the widgets and the BL sits a thin layer of presenters (`MainWindowPresenter`, `GamePresenter`, `BoardPresenter`, `ChatPresenter`): a widget signal becomes a request into the proper BL interface (place, pass, resign, chat), and a BL event becomes a widget update. Presenters never decide legality, they just forward in both directions — and since BL events can arrive from another thread (game loop, network), presenters are also where that gets marshalled back onto the GUI thread.

Neither widgets nor presenters store game state — it all lives behind one interface, `IGameSession`. A local game wraps a core `Game` directly (see [Core.md](Core.md)); a network game talks to a server and mirrors what it's told; hosting just spins up a server in the background and connects to it like any remote client. The GUI only ever talks to `IGameSession`, so switching between local, hosted, and remote never touches a line of widget or presenter code.
