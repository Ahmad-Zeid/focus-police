# WristStream

A watch-only app (no iPhone app) that reads wrist motion at 50 Hz and sends it to the laptop over HTTP every second. It's the live data path for the Focus Police demo, and it can also record training sessions.

```
Apple Watch (WristStream) --HTTP POST every 1 s--> laptop receiver.py --> sessions/LIVE_<time>/stream.csv
                                                                     \--> http://localhost:8000 status page
```

- **What it sends, per sample:** time, acceleration with gravity removed (x/y/z, g), gravity (x/y/z), rotation rate (x/y/z, rad/s), roll/pitch/yaw. These are the same Core Motion values HemiPhysioData records, so the two kinds of recording can be mixed.
- **Screen off:** a watchOS "physical therapy" runtime session keeps the app running with the screen off, for **up to 1 hour** per start. It needs no paid account.
- **Laptop unreachable:** the watch holds up to 10 minutes of data and sends it once the laptop answers again.
- **Phone:** no app on it, but the watch still reaches the network *through* the paired iPhone when they're near each other. That's how watchOS works, not something we chose. Keep the iPhone nearby with Bluetooth and Wi-Fi on.

Built 11 Nov 2026. It compiles for watchOS 8.0, including the Series 3's 32-bit chip, and the receiver was tested with a fake batch. **Not yet tested on the watch.**

## Network: same Wi-Fi, and check campus Wi-Fi early

The watch sends to a **local** address (e.g. `192.168.0.10`), so it only works inside one local network:
- **iPhone nearby:** watch → Bluetooth → iPhone → the iPhone's Wi-Fi → router → laptop.
- **iPhone away:** watch → its own Wi-Fi (Series 3: 2.4 GHz only, networks the iPhone already knows) → router → laptop.

So the **laptop and iPhone must be on the same Wi-Fi**, and the iPhone must not be on cellular only. The laptop's address changes with the network: `receiver.py` prints the current one, and you type it into the watch's Server field.

**Campus networks often block devices from talking to each other** ("client isolation"). Test AUB Wi-Fi well before the demo. If it's blocked, run a separate small network: a travel router, or a hotspot from a second phone, with the laptop and iPhone both joined.

## Which Xcode (11 Nov)

**Xcode 27 can't build for the Series 3** (it supports watchOS 9+ only and has dropped the armv7k chip). **Xcode 16.4** is installed next to it as `/Applications/Xcode-16.4.app`. Its window refuses to open on macOS Tahoe, but its command line works, and that's how the app is built:

```bash
DEVELOPER_DIR=/Applications/Xcode-16.4.app/Contents/Developer xcodebuild -project WristStream.xcodeproj -scheme WristStream -destination 'generic/platform=watchOS' build
```

Built this way it covers all three watch chips (armv7k, arm64_32, arm64). Sign in to your Apple ID in **Xcode 27** (Settings → Accounts); the account is shared with 16.4.

## One-time setup (about 15–30 min, mostly waiting on Xcode)

1. **Sign in to Xcode:** Xcode → Settings → Accounts → **+** → Apple ID. A normal free Apple ID works.
2. **iPhone:** plug it into the Mac with a cable, unlock it, and tap *Trust*. Turn on **Developer Mode** (Settings → Privacy & Security → Developer Mode). The phone restarts.
3. **Watch:** unlock it and keep it near the phone. Xcode finds it through the iPhone. The first time, Xcode spends a while "preparing" the watch.
4. In Xcode (the project is `WristStream.xcodeproj`): click the **WristStream** target → *Signing & Capabilities* → Team = **your name (Personal Team)**.
5. At the top of the Xcode window, choose your **Apple Watch** as the run destination, then press **▶ Run**.
6. If the watch says the developer is untrusted, trust it on the iPhone (Settings → General → VPN & Device Management) and/or on the watch (Settings → General → Profiles or Device Management).

With a free account the app stops opening after **7 days**. Press Run again in Xcode to reinstall it.

## Every time

1. On the laptop:

```bash
cd "WristStream" && python3 receiver.py
```

   - It prints the address the watch should use, e.g. `http://192.168.0.10:8000`. If macOS asks whether Python may accept incoming connections, click **Allow**.
2. On the watch, open WristStream and check the **Server** field matches that address. Tap **Start**.
   - The status should turn green: *Streaming*.
3. Watch it live at http://localhost:8000 (or from the phone's browser at the printed address).
4. To label a live session, put the runner's timestamp file into the new `sessions/LIVE_<time>/` folder and run `segment.py` on it, as with any other session.

## Files

| File | |
|---|---|
| `WristStream/Streamer.swift` | motion reading, background session, batching, HTTP |
| `WristStream/WristStreamApp.swift` | the watch screen: Start/Stop, status, rate, server field |
| `project.yml` | XcodeGen config. Rebuild the project file with `xcodegen generate` after changing it |
| `receiver.py` | laptop receiver: saves the CSV, serves the live page, runs the classifier every 0.5 s. Used by both the Apple and the Galaxy app |
| `live.html` | the live page at http://localhost:8000. See *The live page* below |
| `live_model.py` | the classifier the page shows: `features()`, `load()`, and the on/off-task split. A labelled placeholder until `model.pkl` exists |
| `replay.py` | plays a recorded session into a receiver as if live, to test the page or rehearse the demo without the watch |

## The live page (added 12 Nov)

Open http://localhost:8000 while the receiver runs. It shows:
- **Three live graphs**, scrolling over the last 10 s: motion (acceleration with gravity removed), rotation (gyroscope) and wrist angle (gravity direction). They run about 1.3 s behind real time because the watch sends once a second; that delay keeps the scrolling smooth.
- **The prediction**: the current activity, ON/OFF TASK, the confidence in this guess, every class's probability, a 2-minute on/off strip, and a red **nudge** banner after 10 s off task in a row (`NUDGE_S` in receiver.py).
- **The model's name and its test accuracy.** Confidence (live, per guess) and accuracy (measured once on the held-out test session) are shown separately on purpose.

**Plugging in the trained model:** `../Model/train.py` does it: it saves `model.pkl` next to `live_model.py` (the classifier, its window/sensor/feature settings, the smoothing length, a name, and the on/off-task accuracy on TEST). Features come from `../Model/features.py`, shared by both sides. The receiver loads it within a second, with no restart. Until then a **placeholder** with hand-set thresholds runs, and the page labels it in orange as *not a result*. A model that crashes shows "model error" on the page; recording continues.

**Testing without the watch:** run a second receiver on another port, writing to a scratch folder, and replay a session into it:
```bash
python3 receiver.py 8010 /tmp/rx
```
```bash
python3 replay.py "../Data Collection/sessions/LIVE_20261112-081101/stream.csv" http://localhost:8010
```
Tested 12 Nov this way: graphs, placeholder predictions, hot-loading a dummy `model.pkl` (name and accuracy shown, nudge fired after 10 s off task), falling back to the placeholder when it was removed, and phone width (the prediction moves to the top).

## The session runner is served here too (12 Nov evening)

`http://localhost:8000/runner` serves `../Data Collection/session-runner.html`. Opened this way it shows whether the watch is streaming and warns before a session starts without it. At *finish* it posts its timestamps to `/runner-log`; the receiver saves them into the folder of the recording in progress (`sessions/LIVE_<session>/`), waits 3 s for the last claps to arrive, runs `segment.py`, and returns its output and `check.png` to the page. An earlier runner file in that folder is kept as `*.replaced`. Tested with a replayed session on a test receiver.
