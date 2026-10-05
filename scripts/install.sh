#!/usr/bin/env bash
# Install whisper-service as a macOS LaunchAgent: it starts at login, restarts
# if it crashes, and listens on 127.0.0.1 only. Then download and warm up the
# model, so the very first dictation is instant. Safe to run again after a
# `git pull` - it re-syncs the dependencies and restarts the service.
#
#   scripts/install.sh            install or update, then restart
#   scripts/uninstall.sh          stop and remove the LaunchAgent

set -euo pipefail

LABEL="com.gapsong.whisper-service"
HOST="127.0.0.1"
PORT="9876"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd -P)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG="$HOME/Library/Logs/whisper-service.log"
DOMAIN="gui/$(id -u)"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "whisper-service needs macOS on Apple Silicon (MLX)." >&2
  exit 1
fi
# uv installs Python and the locked dependencies. A fresh Mac has neither uv nor
# Homebrew, so fall back to uv's official installer. It puts uv into
# ~/.local/bin; UV_NO_MODIFY_PATH keeps it from editing shell profiles, so this
# script calls uv by its full path instead.
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV="$HOME/.local/bin/uv"
fi
if [ -z "$UV" ]; then
  if command -v brew >/dev/null; then
    echo "==> installing uv (Homebrew)"
    brew install uv
    UV="$(command -v uv)"
  else
    echo "==> installing uv (official installer, into ~/.local/bin)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
    UV="$HOME/.local/bin/uv"
  fi
fi

# .python-version pins the Python that every dependency has a prebuilt wheel
# for; uv downloads it if the Mac does not have it.
echo "==> installing dependencies (uv.lock)"
(cd "$REPO_DIR" && "$UV" sync --frozen --no-dev --compile-bytecode)

echo "==> writing $PLIST"
mkdir -p "$(dirname "$PLIST")" "$(dirname "$LOG")"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$REPO_DIR/.venv/bin/python</string>
    <string>-m</string>
    <string>uvicorn</string>
    <string>--factory</string>
    <string>app.server:create_default_app</string>
    <string>--host</string>
    <string>$HOST</string>
    <string>--port</string>
    <string>$PORT</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$REPO_DIR</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>$LOG</string>
  <key>StandardErrorPath</key>
  <string>$LOG</string>
</dict>
</plist>
EOF

echo "==> (re)starting $LABEL"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
# bootout returns before launchd has finished removing the service; an
# immediate bootstrap then fails with "5: Input/output error". Wait it out.
for _ in $(seq 1 50); do
  launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || break
  sleep 0.2
done
launchctl bootstrap "$DOMAIN" "$PLIST"

URL="http://$HOST:$PORT"
up=0
for _ in $(seq 1 120); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 0.5
done
if [ "$up" != 1 ]; then
  echo "service did not come up - see $LOG" >&2
  exit 1
fi
echo "==> running: $URL  (log: $LOG)"

# Load the model now. The first time this downloads it (~1.6 GB) into
# ~/.cache/huggingface; later runs only load it from disk (well under a second).
echo "==> loading the model (the first time downloads ~1.6 GB, give it a few minutes)"
log_offset=$(wc -c < "$LOG")
curl -fsS -X POST "$URL/start" >/dev/null
waited=0
until curl -fsS "$URL/health" 2>/dev/null | grep -q '"ready":true'; do
  # A failed load (e.g. no internet for the download) drops the service back
  # to sleeping and logs why; stop here instead of waiting for nothing.
  if tail -c +"$((log_offset + 1))" "$LOG" | grep -q "loading the whisper model failed"; then
    echo "loading the model failed - the reason is at the end of $LOG" >&2
    exit 1
  fi
  if [ "$waited" -ge 1800 ]; then
    echo "model not ready after 30 minutes - see $LOG" >&2
    exit 1
  fi
  if [ "$waited" -gt 0 ] && [ $((waited % 15)) -eq 0 ]; then
    echo "    still loading... ${waited}s"
  fi
  sleep 1
  waited=$((waited + 1))
done
echo "==> ready: whisper-service answers on $URL"
echo
echo "Next: install the Mac app - https://github.com/gapsong/mac-voice-dictation"
