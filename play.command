#!/usr/bin/env bash
# ==============================================================================
#  W(x,y) Hyrule — double-click to host the game and play (macOS / Linux).
#  Starts the multiplayer game server (server.mjs), opens the game in your
#  browser, and — with cloudflared installed — prints an https link to send
#  to friends anywhere (they need only Chrome or Edge, on Windows or a Mac).
#  Windows: double-click play.bat instead.
# ==============================================================================
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=============================================================================="
echo "                W(x,y) HYRULE — multiplayer game server                       "
echo "=============================================================================="

if ! command -v node > /dev/null 2>&1; then
  echo "Node.js is needed: install it from https://nodejs.org (the LTS version), then double-click play.command again."
  read -n 1 -s -r -p "Press any key to close…"
  exit 1
fi

# Already running (another window): just open the game.
if [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/mp || true)" = "426" ]; then
  echo "The game server is already running: opening http://localhost:3000/game"
  open http://localhost:3000/game
  exit 0
fi

if [ ! -d node_modules ]; then
  echo "[setup] Installing the game's packages (first run)…"
  if command -v bun > /dev/null 2>&1; then bun install; else npm install; fi
fi

exec node server.mjs --prod --tunnel --open
