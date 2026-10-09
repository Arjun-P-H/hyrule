@echo off
rem ==============================================================================
rem  W(x,y) Hyrule - double-click to host the game and play (Windows).
rem  Starts the multiplayer game server (server.mjs), opens the game in your
rem  browser, and - with cloudflared installed - prints an https link to send
rem  to friends anywhere (they need only Chrome or Edge, on Windows or a Mac).
rem  Friends who join do NOT need this: they just open the link.
rem ==============================================================================
setlocal
cd /d "%~dp0"
title W(x,y) Hyrule - game server
echo ==============================================================================
echo                 W(x,y) HYRULE - multiplayer game server
echo ==============================================================================

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is needed: install it from https://nodejs.org ^(the LTS version^), then run play.bat again.
  pause
  exit /b 1
)

rem Already running (another window): just open the game.
set CODE=
for /f %%c in ('curl -s -o NUL -w "%%{http_code}" http://localhost:3000/mp 2^>NUL') do set CODE=%%c
if "%CODE%"=="426" (
  echo The game server is already running: opening http://localhost:3000/game
  start "" http://localhost:3000/game
  exit /b 0
)

if not exist node_modules (
  echo [setup] Installing the game's packages ^(first run^)...
  call npm install
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

node server.mjs --prod --tunnel --open
pause
