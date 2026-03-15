# Tengen Application

## Architecture

The Business logic is handled through the SessionManager.
The GUI is split into presenter and widget classes.
Inside the GUI layer, we use Qt signals to communicate.

The Widget classes just render data and emit signals on user input.
The Presenter class of each widget subscribes to the relevant events from the SessionManager and handles signals emitted from the corresponding Widget class.

So the architecture is:
User -> Wiget -> Presenter -> SessionManager


Keep the Widgets dumb and in the game::gui library so we can reuse them for other projects (like tools).