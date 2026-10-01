#!/usr/bin/env bash
# Install whisper-service as a macOS LaunchAgent: it starts at login, restarts
# if it crashes, and listens on 127.0.0.1 only. Safe to run again after a
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
command -v uv >/dev/null || { echo "uv is missing: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "==> installing dependencies (uv.lock)"
(cd "$REPO_DIR" && uv sync --frozen --no-dev --compile-bytecode)

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

for _ in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "==> running: http://$HOST:$PORT  (log: $LOG)"
    curl -fsS "http://$HOST:$PORT/health"; echo
    exit 0
  fi
  sleep 0.5
done
echo "service did not come up - see $LOG" >&2
exit 1
