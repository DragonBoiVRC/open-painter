# Launcher and automatic updates

`packaging/install.sh` installs a small launcher script as `~/.local/bin/open-painter` and the real
program as `~/.local/share/open-painter/open-painter-bin`. Starting the launcher:

1. starts the program straight away;
2. in the background, at most once every six hours, runs `git fetch` in the source checkout recorded in
   `~/.local/share/open-painter/source`;
3. if the remote has new commits and the checkout has no uncommitted changes, fast-forwards it,
   rebuilds with `packaging/install.sh --update` and swaps the new binary in atomically.

The new version is used the next time the program starts. Everything is logged to
`~/.local/state/open-painter/update.log`; a desktop notification reports success or failure.

Switch it off with `off` in `~/.config/open-painter/auto-update` or `OPEN_PAINTER_NO_UPDATE=1`.
Updating needs git access to the repository and the Rust toolchain; it never touches a checkout with
local changes or one that has commits the remote does not.
