#!/usr/bin/env bash
# Stop whisper-service and remove its LaunchAgent. Leaves the repo, the
# virtualenv and the downloaded model (~/.cache/huggingface) in place.

set -euo pipefail

LABEL="com.gapsong.whisper-service"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "==> removed $LABEL"
