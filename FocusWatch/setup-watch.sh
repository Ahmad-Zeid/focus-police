#!/bin/bash
# One-time setup of the Galaxy Watch, then a 15-second live test.
#   bash setup-watch.sh              # real watch over Wi-Fi (asks for the pairing code)
#   bash setup-watch.sh --emulator   # the Wear OS emulator (for testing)
#   bash setup-watch.sh --uninstall  # remove the app from the watch (give it back clean)
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ADB="$(command -v adb || echo "$HOME/Library/Android/sdk/platform-tools/adb")"
PKG="edu.aub.focuspolice.watch"
APK="$HERE/FocusPolice-watch.apk"
RECEIVER="$HERE/../WristStream/receiver.py"
PY="${PYTHON:-$(command -v python3)}"      # set PYTHON=/path/to/venv/bin/python to use a virtualenv
PORT=8000
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31mPROBLEM: %s\033[0m\n' "$*"; exit 1; }

[ -x "$ADB" ] || die "adb not found at $ADB"
[ -f "$APK" ] || die "App file missing: $APK"

# ------------------------------------------------------------ connect
if [ "${1:-}" = "--emulator" ]; then
  DEV=$("$ADB" devices | awk '/^emulator-/{print $1; exit}')
  [ -n "$DEV" ] || die "No emulator running."
  SERVER="http://10.0.2.2:$PORT"          # how the emulator reaches the Mac
else
  DEV=$("$ADB" devices | awk 'NR>1 && $2=="device" && $1 !~ /^emulator-/ {print $1; exit}')
  if [ -z "$DEV" ]; then
    say "Watch: Settings > Developer options > Wireless debugging > 'Pair new device'."
    read -r -p "Pairing IP:port shown on the watch (e.g. 192.168.1.50:41234): " PAIR
    read -r -p "Pairing code: " CODE
    "$ADB" pair "$PAIR" "$CODE" || die "Pairing failed. Same Wi-Fi as the laptop? Code still on screen?"
    say "Now the IP address & port on the main Wireless debugging screen (NOT the pairing one)."
    read -r -p "IP:port: " CONN
    "$ADB" connect "$CONN" | tee /dev/stderr | grep -q "connected" || die "Could not connect to $CONN"
    DEV="$CONN"
  fi
  IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
  [ -n "$IP" ] || die "The Mac has no Wi-Fi IP address."
  SERVER="http://$IP:$PORT"
fi
A() { "$ADB" -s "$DEV" "$@"; }
say "Device: $DEV  ($(A shell getprop ro.product.model | tr -d '\r'), Android $(A shell getprop ro.build.version.release | tr -d '\r'), API $(A shell getprop ro.build.version.sdk | tr -d '\r'))"

if [ "${1:-}" = "--uninstall" ] || [ "${2:-}" = "--uninstall" ]; then
  A uninstall "$PKG" && say "Removed. The watch is as it was."; exit 0
fi

# ------------------------------------------------------------ install + keep alive
say "Installing the app…"
A install -r -g "$APK" | tail -1 | grep -q Success || die "Install failed."
A shell pm grant "$PKG" android.permission.POST_NOTIFICATIONS 2>/dev/null
A shell dumpsys deviceidle whitelist +"$PKG" >/dev/null           # never paused by Doze
A shell cmd appops set "$PKG" RUN_IN_BACKGROUND allow 2>/dev/null
A shell cmd appops set "$PKG" RUN_ANY_IN_BACKGROUND allow 2>/dev/null
A shell am set-standby-bucket "$PKG" active 2>/dev/null
A shell dumpsys deviceidle whitelist | grep -q "$PKG" && echo "battery-saver exemption: ON" || echo "WARNING: exemption not confirmed"

# ------------------------------------------------------------ receiver
if ! curl -s -m 2 "http://localhost:$PORT/status" >/dev/null; then
  say "Starting the laptop receiver (log: /tmp/focus-receiver.log)…"
  nohup "$PY" "$RECEIVER" "$PORT" > /tmp/focus-receiver.log 2>&1 &
  sleep 2
  curl -s -m 2 "http://localhost:$PORT/status" >/dev/null || die "Receiver did not start; see /tmp/focus-receiver.log"
fi

# ------------------------------------------------------------ live test
say "Setting the laptop address on the watch to $SERVER and running a 15 s test…"
A shell am start -n "$PKG/.MainActivity" --es server "$SERVER" >/dev/null
sleep 2
BEFORE=$(curl -s "http://localhost:$PORT/status" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["total"])')
A shell am start -n "$PKG/.MainActivity" --ez autostart true >/dev/null
sleep 15
A shell am start -n "$PKG/.MainActivity" --ez autostop true >/dev/null
sleep 4
STATUS=$(curl -s "http://localhost:$PORT/status")
AFTER=$(echo "$STATUS" | "$PY" -c 'import sys,json;print(json.load(sys.stdin)["total"])')
GOT=$((AFTER - BEFORE))
echo "$STATUS"
if [ "$GOT" -ge 500 ]; then
  say "PASS: received $GOT samples in ~15 s (expect ~750). Ready."
  say "Status page: http://localhost:$PORT    Sessions: Project/Data Collection/sessions/LIVE_*"
else
  printf '\n\033[31mFAIL: only %s samples arrived.\033[0m\n' "$GOT"
  echo "Check: watch and Mac on the same Wi-Fi? macOS asked to allow Python incoming connections -> Allow."
  echo "Watch screen shows the reason under the button. Laptop address on watch: $SERVER"
  exit 1
fi
