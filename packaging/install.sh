#!/bin/sh
# Installs open-painter for the current user: the program, a launcher that keeps it up to date,
# an application-menu entry with an icon, and an association for .opproj files.
# Nothing outside your home directory is touched.
#
#   packaging/install.sh             build (if needed) and install everything
#   packaging/install.sh --update    rebuild and replace only the program (used by the launcher)
#   packaging/install.sh --remove    take it all out again
set -eu

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
data="${XDG_DATA_HOME:-$HOME/.local/share}/open-painter"
conf_dir="${XDG_CONFIG_HOME:-$HOME/.config}/open-painter"
bin_dir="$HOME/.local/bin"
apps_dir="$HOME/.local/share/applications"
icon_dir="$HOME/.local/share/icons/hicolor/scalable/apps"
png_dir="$HOME/.local/share/icons/hicolor/256x256/apps"
mime_dir="$HOME/.local/share/mime"
mode="${1:-install}"

refresh() {
    command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$apps_dir" || true
    command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$mime_dir" || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "$HOME/.local/share/icons/hicolor" || true
    command -v kbuildsycoca6 >/dev/null 2>&1 && kbuildsycoca6 >/dev/null 2>&1 || true
}

if [ "$mode" = "--remove" ]; then
    rm -f "$bin_dir/open-painter" "$apps_dir/open-painter.desktop" "$icon_dir/open-painter.svg" \
          "$png_dir/open-painter.png" "$mime_dir/packages/open-painter.xml"
    rm -rf "$data" "$conf_dir"
    refresh
    echo "open-painter removed."
    exit 0
fi

# Build the optimised program if it is missing or older than the sources.
export PATH="$HOME/.cargo/bin:$PATH"
bin="$root/target/release/op-app"
if [ ! -x "$bin" ] || [ -n "$(find "$root/crates" -name '*.rs' -newer "$bin" -print -quit)" ] || [ "$mode" = "--update" ]; then
    [ "$mode" = "--update" ] || echo "Building open-painter (this takes a few minutes the first time)..."
    (cd "$root" && cargo build --release -p op-app)
fi

mkdir -p "$data"
# Replace the program atomically, so a running copy is never disturbed.
install -m 755 "$bin" "$data/open-painter-bin.new"
mv -f "$data/open-painter-bin.new" "$data/open-painter-bin"
[ "$mode" = "--update" ] && exit 0

printf '%s\n' "$root" > "$data/source"
mkdir -p "$conf_dir" "$bin_dir" "$apps_dir" "$icon_dir" "$png_dir" "$mime_dir/packages"
[ -e "$conf_dir/auto-update" ] || echo on > "$conf_dir/auto-update"
install -m 755 "$here/open-painter-launcher" "$bin_dir/open-painter"
install -m 644 "$here/open-painter.svg" "$icon_dir/open-painter.svg"
install -m 644 "$here/open-painter.png" "$png_dir/open-painter.png"
install -m 644 "$here/open-painter-mime.xml" "$mime_dir/packages/open-painter.xml"
# The menu entry names the launcher by its full path, so it works even if ~/.local/bin is not
# on the PATH of your desktop session.
sed "s|@EXEC@|$bin_dir/open-painter|" "$here/open-painter.desktop" > "$apps_dir/open-painter.desktop"
chmod 644 "$apps_dir/open-painter.desktop"
refresh

echo "Installed. Look for 'open-painter' in your application menu."
echo "From a terminal: $bin_dir/open-painter [model-or-project]"
echo "Automatic updates are on (checked at most every 6 hours); see README.md to turn them off."
case ":$PATH:" in *":$bin_dir:"*) ;; *) echo "Note: $bin_dir is not on your PATH, so only the menu entry will find it." ;; esac
