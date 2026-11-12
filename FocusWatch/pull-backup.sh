#!/bin/bash
# Copies every session CSV the watch saved locally (the backup copy) into
# Project/Data Collection/sessions/WATCH_BACKUP/. Use if streaming ever failed.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ADB="$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")"
PKG="edu.aub.focuspolice.watch"
OUT="$HERE/../Data Collection/sessions/WATCH_BACKUP"
DEV=$("$ADB" devices | awk 'NR>1 && $2=="device" {print $1; exit}')
[ -n "$DEV" ] || { echo "No watch connected (run setup-watch.sh first)."; exit 1; }
mkdir -p "$OUT"
for f in $("$ADB" -s "$DEV" shell run-as "$PKG" ls files/sessions 2>/dev/null | tr -d '\r'); do
  "$ADB" -s "$DEV" exec-out run-as "$PKG" cat "files/sessions/$f" > "$OUT/$f"
  echo "pulled $f ($(wc -l < "$OUT/$f") lines)"
done
