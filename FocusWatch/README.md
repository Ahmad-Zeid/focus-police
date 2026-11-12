# Focus Police: Galaxy Watch app

A Wear OS app for the borrowed **Galaxy Watch5 Pro**. It records wrist motion at 50 Hz and streams it to the laptop over Wi-Fi. No phone is involved and no account is needed. The same app covers both jobs:

- **Recording training data:** Start, do the session with the runner, Stop. The laptop receiver saves it as `Data Collection/sessions/LIVE_<time>/stream.csv`.
- **The live demo / focus session:** the same Start button. The laptop does the live classification: `receiver.py` runs `../WristStream/live_model.py` and shows live graphs plus the prediction at http://localhost:8000 (see `../WristStream/README.md`, *The live page*). The watch doesn't need to know which of the two it's doing.

```
Galaxy Watch (Focus Police) --Wi-Fi, HTTP POST every 1 s--> laptop receiver.py --> sessions/LIVE_<time>/stream.csv
          \--> backup CSV on the watch itself (pull-backup.sh)       \--> http://localhost:8000 live page: graphs + prediction
```

## How it stays alive (the Apple Watch problem, handled)

| Risk | What the app does |
|---|---|
| The system pauses the app when the screen goes off | Runs as a **foreground service** (the Android equivalent of a workout session), with a permanent notification and a **partial wake lock** so the processor stays awake |
| Doze / battery saver | `setup-watch.sh` adds the app to the system's **battery-optimisation whitelist** and allows background running |
| The watch turns Wi-Fi off, or routes through a phone | The app explicitly **requests Wi-Fi** from the system for as long as it records, and falls back to any network |
| Laptop briefly unreachable | Samples **queue on the watch** (up to 30 min) and are sent when it's back. Nothing is dropped |
| Laptop down for good, or a crash | Every sample is also written to a **CSV on the watch**. `pull-backup.sh` copies them off |
| The app gets killed anyway | The service is "sticky": Android restarts it and it resumes recording (new file, same day) |

## Data

16 columns per sample: `time` (epoch), `userAcceleration{X,Y,Z}` (g, gravity removed), `gravity{X,Y,Z}` (g), `rotationRate{X,Y,Z}` (rad/s), `roll, pitch, yaw` (rad), `acc{X,Y,Z}` (raw accelerometer, g). The first 13 have the same names and units as the Apple Watch apps, so `segment.py` reads both.

Note: Android's gravity vector points **up** and Apple's points **down**. That only matters when mixing Apple and Galaxy data. Training on Galaxy data only, as planned, avoids it.

## Tomorrow: see `WATCH-SETUP.md` next to this file.

## Server mode (only if the laptop route fails, e.g. on AUB Wi-Fi)

The watch sends to whatever address is in its **Laptop address** field, so a server needs **no change to the app**:

1. On the server, copy `../WristStream/receiver.py` (Python 3, standard library only, no installs) and run it:
   `python3 receiver.py 8000 ./sessions`. The second argument is where sessions get saved.
2. Open that port on the server's firewall, or put it behind the server's existing web server or HTTPS reverse proxy (path `/ingest` and `/status`).
3. On the watch, type the address into the field: `https://your-domain` (behind HTTPS, no port needed) or `http://your-domain:8000`. Or set it from the laptop: `adb shell am start -n edu.aub.focuspolice.watch/.MainActivity --es server https://your-domain`.
4. The watch then needs **internet**, not the same network as the laptop. Get the recordings with `scp`, or watch them live on `https://your-domain/status`.

Notes:
- `setup-watch.sh` always sets the address back to the laptop, so set the server address **after** running it.
- A server reachable only through a VPN can't be used: the watch can't join one.
- Address handling tested 12 Nov: `192.168.1.50` → `http://192.168.1.50:8000`; `https://example.com` stays as typed.

## Files

| File | |
|---|---|
| `FocusPolice-watch.apk` | the built app, ready to install |
| `setup-watch.sh` | pair, install, battery exemption, set the laptop address, 15 s live test. `--uninstall` removes the app |
| `pull-backup.sh` | copy the watch-side backup CSVs to `Data Collection/sessions/WATCH_BACKUP/` |
| `app/src/main/java/.../SensorService.kt` | sensors, foreground service, Wi-Fi, queue, upload, backup file |
| `app/src/main/java/.../MainActivity.kt` | the watch screen |
| `build.sh` | rebuild the APK (JDK 17 + Android SDK in `~/Library/Android/sdk`) |
