# Galaxy Watch setup (written 12 Nov as a one-day checklist)

Total: about 15 min of setup, then the recording sessions.

## Bring

- The watch, **charged**, and **its charger**. Streaming uses more battery than normal; charge between sessions.
- The laptop and the watch on **the same Wi-Fi**. Home Wi-Fi is safest; campus Wi-Fi often blocks devices from talking to each other.

## 1. On the watch (5 min, once)

*Sources checked 12 Nov: Google's "Debug Wear OS over Wi-Fi" page and Samsung's own "Connect Galaxy Watch to Android Studio over Wi-Fi" guide. Menu names can differ slightly between watch software versions.*

1. **Wi-Fi:** Settings → Connections → Wi-Fi → join the same network as the laptop. You type the Wi-Fi password on the watch. Keep **Power saving** off.
2. **Developer options:** Settings → **About watch** → **Software** (may be called *Software information*) → tap **Software version** repeatedly, about 5–7 times, until it says developer mode is on. (Samsung guides say 5; others say 7. Keep tapping until the message appears.)
3. Settings → **Developer options**:
   - turn on **ADB debugging**;
   - turn on **Turn off automatic Wi-Fi**. Samsung's guide recommends this so the watch doesn't drop Wi-Fi;
   - turn on **Wireless debugging**, and choose **Always allow on this network** in the prompt;
   - inside Wireless debugging, tap **Pair new device**. The watch shows a **Wi-Fi pairing code, IP address and port**. Keep that screen open.
   - The watch uses **two different ports**: one under *Pair new device* (for pairing) and one on the main *Wireless debugging* screen (for connecting). The script asks for both.

## 2. On the laptop (one command)

```bash
cd "FocusWatch" && bash setup-watch.sh
```

It asks for:
- the pairing IP:port and code;
- then the IP:port on the **main** Wireless debugging screen (a different port).

Then it installs the app, turns off battery saving for it, starts the receiver, sets the laptop address on the watch, and runs a **15-second test**. It ends with **PASS** or **FAIL** plus the reason.

- If the watch asks to "Allow debugging", tap **Allow / Always allow**.
- The macOS firewall is off on this Mac (checked 12 Nov), so no "allow incoming connections" prompt is expected.

## Already tested (12 Nov, emulators: Wear OS 5 = Android 14 / API 34, and Wear OS 6 = Android 16 / API 36)

| Test | Wear OS 5 | Wear OS 6 |
|---|---|---|
| `setup-watch.sh` end to end | PASS, 751 samples in 15 s | PASS, 751 samples in 15 s |
| Screen off + forced deep Doze | 50 Hz kept | 50 Hz kept |
| Laptop receiver killed for 30 s mid-session | no gap, backlog delivered | no gap, backlog delivered |
| App process killed | restarted in about 1 s, new session | restarted, new session |
| 100 s session completeness | 4991/4996 samples, largest gap 0.13 s | 4984/4986, largest gap 0.064 s |
| Watch-side backup (`pull-backup.sh`) | had every sample, including 9 unsent at the kill | — |
| Full pipeline (claps → runner clicks → `segment.py`) | 3/3 claps found, both activities labelled | — |
| Crashes / background-start errors in the system log | none | none |

Test sessions were moved to `../Data Collection/sessions/_emulator-tests-12Nov/` so the real recordings start clean.

Not testable without the real watch: pairing over wireless debugging, real Wi-Fi, Samsung's own battery management, real sensor noise. `setup-watch.sh`'s 15 s test checks all of that in one go.

## 3. Each recording session

1. Receiver running (setup leaves it running; to start it by hand: `python3 ../WristStream/receiver.py`). Live page with graphs and the (placeholder) prediction: http://localhost:8000
2. Open **http://localhost:8000/runner** on the second screen and fill in the session details. It shows whether the watch is streaming. (Updated 12 Nov evening: the receiver serves the runner now.)
3. On the watch: open **Focus Police** → **START**. The screen should say *laptop connected*. You can let the screen go dark; it keeps recording.
4. Run the session (clap 3×, click, and so on). At the end: **STOP** on the watch.
5. Nothing to move: when you click *finish*, the runner hands its timestamps to the receiver, which files them in the session's `sessions/LIVE_<time>/` folder, runs `segment.py`, and shows the result and `check.png` on the page.
6. **New network?** The watch keeps the old laptop address. Open Focus Police on the watch, tap the address, and type the laptop's current one (`ipconfig getifaddr en0` on the laptop prints it; the app adds `http://` and `:8000`).

Plan (12 Nov evening): PILOT → S1 → S2 → TEST → S3 → NATURAL → OTHERHAND. The design and reasons are in `../Data Collection/README.md`, section 0.

## If something goes wrong

| Symptom | Fix |
|---|---|
| Watch says "laptop NOT reached" | Same Wi-Fi? Is the receiver running? Is the address under the button the laptop's current IP? (`setup-watch.sh` sets it.) Recording continues on the watch either way |
| Wireless debugging won't pair | Toggle Wireless debugging off and on, tap Pair again. A Samsung update made this flaky for some people; retrying usually works |
| Session missing on the laptop | `bash pull-backup.sh` copies the watch's backup copies |

## Giving the watch back

```bash
bash setup-watch.sh --uninstall
```

Then on the watch, turn off Developer options (Settings → Developer options → off).
