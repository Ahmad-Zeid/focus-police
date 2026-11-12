#!/bin/bash
# Rebuilds FocusPolice-watch.apk. Build files go to ~/Library/Caches/FocusWatch (outside the project folder).
set -euo pipefail
cd "$(dirname "$0")"
# Needs JDK 17 and the Android SDK (platform 35, build-tools 35). Override these two if yours live elsewhere.
export JAVA_HOME="${JAVA_HOME:-/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home}"
export ANDROID_HOME="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
./gradlew --project-cache-dir "$HOME/Library/Caches/FocusWatch/gradle" -q assembleDebug
cp "$HOME/Library/Caches/FocusWatch/app/outputs/apk/debug/app-debug.apk" FocusPolice-watch.apk
echo "built: $(pwd)/FocusPolice-watch.apk"
