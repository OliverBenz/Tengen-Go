# Bot Engines

Bot games are played against [GNU Go](https://www.gnu.org/software/gnugo/) or [KataGo](https://github.com/lightvector/KataGo).
They run as separate processes and are not part of this repository, so you have to provide them yourself.

Tengen looks for them in an `engine` folder, first in the user's data folder, then next to its own executable (for a Windows debug build, that is `build/Win64/out/bin/x64/Debug/`).
Each engine comes from the first folder that holds all of its files, and engines in the data folder serve every build.

| OS      | Data folder                                   |
| ------- | --------------------------------------------- |
| Linux   | `~/.local/share/tengen/engine`                |
| Windows | `%LOCALAPPDATA%\tengen\engine`                |
| macOS   | `~/Library/Application Support/tengen/engine` |

```
engine/
├── gnugo/
│   └── gnugo(.exe)
└── katago/
    ├── katago(.exe)
    ├── model.bin.gz
    ├── human_model.bin.gz
    └── gtp.cfg
```

The bot dialog lists both engines. An engine whose files are not all there is greyed out. The dialog checks again every time it opens, so an engine installed while Tengen runs shows up right away.

## GNU Go

On Windows, download a GNU Go 3.8 build and copy its whole folder into `engine/gnugo/`, not just `gnugo.exe`: builds like the Cygwin one need their DLLs next to the executable.

On Linux, install GNU Go through your package manager and link it into place:

```
mkdir -p ~/.local/share/tengen/engine/gnugo
ln -s "$(command -v gnugo)" ~/.local/share/tengen/engine/gnugo/gnugo
```

GNU Go's strength is a level from 1 to 10, and that is what the bot dialog offers.
The levels are not ranks: GNU Go plays around 5k to 8k at level 10, and the lower levels play weaker by an amount nobody has measured.

## KataGo

Copy KataGo's whole folder into `engine/katago/`, then add:

- a KataGo network, as `model.bin.gz`
- the human SL network `b18c384nbt-humanv0.bin.gz` from [katagotraining.org's extra networks](https://katagotraining.org/extra_networks/), as `human_model.bin.gz`
- a GTP config made for the human SL network, like KataGo's `gtp_human5k_example.cfg`, as `gtp.cfg`

KataGo imitates a human of the rank you pick in the bot dialog, from 20k to 3d.
