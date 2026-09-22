#!/usr/bin/env bash
set -e

# ==============================================================================
#  🎮 W(x,y) Hyrule Continuous World Engine — Browser Game Launcher
#  WebGL2 Procedural Terrain Engine · 120-240 FPS
# ==============================================================================

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=============================================================================="
echo "          🎮 W(x,y) HYRULE CONTINUOUS WORLD ENGINE LAUNCHER                   "
echo "                     WebGL2 Browser Engine · 120-240 FPS                      "
echo "=============================================================================="
echo ""

# Check if Next.js local server is already running on port 3000
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 || true)
if [[ "$HTTP_CODE" =~ ^(200|304|307|308)$ ]]; then
  echo "[1/2] Local engine server already active on http://localhost:3000."
else
  echo "[1/2] Starting local game server on http://localhost:3000 in background..."
  if command -v bun > /dev/null 2>&1; then
    bun run dev > /dev/null 2>&1 &
  else
    npm run dev > /dev/null 2>&1 &
  fi
  echo "Waiting for engine server to initialize..."
  
  MAX_ATTEMPTS=40
  ATTEMPT=0
  SERVER_UP=false
  while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
      SERVER_UP=true
      break
    fi
    sleep 0.5
    ATTEMPT=$((ATTEMPT + 1))
  done

  if [ "$SERVER_UP" = true ]; then
    echo "Engine server is ready!"
  else
    echo "⚠️  Server did not respond within 20s. Opening browser anyway..."
  fi
fi

echo "[2/2] Opening game in your browser..."
open http://localhost:3000/game

echo ""
echo "Game launched at http://localhost:3000/game!"
