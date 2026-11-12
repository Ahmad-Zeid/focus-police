# Focus Police: full handoff

Written 13 Nov 2026 as a record of what was built, how, and why, for the rest of the team and any AI assistant they work with. `README.md` is the short version.

Everything here was built between 11 and 13 Nov 2026 by one team member working with an AI coding agent. The final submission is due **Tue 17 Nov 2026, 23:55**.

---

## 1. Status in one screen

**The project.** Focus Police: a watch on the wrist streams motion to a laptop; a classifier decides which desk activity is happening and whether it is on task (working) or off task (phone); after 10 s off task the laptop shows a nudge.

**What works, and how well it was checked:**

| Piece | State | Checked how |
|---|---|---|
| Watch app (Galaxy Watch5 Pro, Wear OS) streaming 50 Hz to the laptop | Works | Real watch, several recordings, no gaps; stress-tested in two emulators |
| Laptop receiver + live page (graphs, prediction, nudge) | Works | Real watch and replayed recordings |
| Session runner (tells you what to do, logs timestamps) | Works | Real sessions |
| Automatic labelling at the end of a session | Works | One real session, emulator sessions |
| Training pipeline (210 model combinations, report, deploy to live page) | Runs | Synthetic sessions and **one** real session |
| **A model worth reporting** | **No** | See below |

**State of the model.** There is one labelled recording: 8 activities, one block of about a minute each, one person, one sitting. There is no test session yet. The model trained on it scores 77% (activity) and 98% (phone vs work) when a quarter of each block is held out; that number is optimistic, because the held-out part comes from the same minute as the training part.

**Live trial (12 Nov, right after training):** not great. Holding the phone was mistaken for drinking, and typing for writing, among others. It looks like more data is needed.

**The watch.** So far there is one Galaxy Watch5 Pro, borrowed. The app installs on any Wear OS 3+ watch with `FocusWatch/setup-watch.sh`, so a second watch can record too. Without a watch, everything else still runs: labelling, training, analysis, and replaying recordings into the live page.

---

## 2. What is in this folder

```
README.md                 short human overview
HANDOFF.md                this file
AGENTS.md                 pointer for AI assistants
requirements.txt          Python packages (versions the saved model was built with)

FocusWatch/               the Wear OS watch app (Kotlin) + setup scripts
  FocusPolice-watch.apk     built app, ready to install
  setup-watch.sh            pair, install, exempt from battery saving, set laptop address, 15 s test
  pull-backup.sh            copy the watch's own backup CSVs to the laptop
  build.sh                  rebuild the APK
  WATCH-SETUP.md            step-by-step for the person holding the watch
  app/src/main/...          SensorService.kt (everything important), MainActivity.kt (the one screen)

WristStream/              the LAPTOP side. (The name is historical: it began as an Apple Watch app.)
  receiver.py               HTTP server: receives watch data, saves CSV, serves the pages, runs the model
  live.html                 the live page (graphs + prediction + nudge)
  live_model.py             loads model.pkl and predicts; placeholder rules if there is no model
  model.pkl                 the current trained model (see section 9 for what is inside)
  replay.py                 play a recording into the receiver as if the watch were sending it
  WristStream/, *.xcodeproj, project.yml    the abandoned Apple Watch app (section 11). Not used.

Data Collection/
  session-runner.html       the runner page (served by the receiver at /runner)
  segment.py                turns one recording + its runner timestamps into labelled data
  sessions/                 every real recording (section 8), one folder each
  README.md                 the protocol and its history
  u01_misc_...csv, pilot-1331-typing-check.png   an early Apple Watch test recording (history only)

Model/
  features.py               window -> numbers. Shared by training AND the live page
  train.py                  the whole experiment
  REPORT.md                 latest report
  results/<time>/           every run: report, all_combinations.csv, figures
  README.md
```

The four top-level folders find each other by relative path (`../Data Collection/sessions`, `../Model`, `../WristStream/model.pkl`). Renaming one means updating those paths. The folder names contain spaces, so they need quoting in shell commands.

---

## 3. Quick start without a watch

Python 3.10 or newer.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Train on whatever is in `Data Collection/sessions/`** (under a minute today, about 4 minutes with four full sessions):

```bash
cd Model && python3 train.py
```

It prints the top combinations, writes `Model/REPORT.md` and `Model/results/<time>/`, and overwrites `WristStream/model.pkl`. Use `--no-deploy` to leave the live model alone, `--quick` for a one-minute sanity run.

**See the live page with a recording instead of a watch.** Two terminals:

```bash
cd WristStream && python3 receiver.py 8010 /tmp/focus-test
```

```bash
cd WristStream && python3 replay.py "../Data Collection/sessions/LIVE_20261112-233400/stream.csv" http://localhost:8010
```

Then open http://localhost:8010. (`localhost` means "this computer": the pages are served by the receiver running on your own laptop, so they work for whoever runs it and are not a link to anyone else's machine.) The second argument to `receiver.py` is where it saves; pointing it at a scratch folder keeps replays out of `sessions/`. `replay.py --from 250` starts 250 s into the recording.

---

## 4. How the system fits together

```
Galaxy Watch (Focus Police app)
   │  50 Hz samples, batched once a second, HTTP POST /ingest over Wi-Fi (no phone involved)
   ▼
receiver.py on the laptop (port 8000)
   ├─ appends to  Data Collection/sessions/LIVE_<start time>/stream.csv   (the raw recording)
   ├─ keeps the last 15 s in memory
   ├─ every 0.5 s: live_model.predict(last WINDOW seconds) -> label, probabilities, on/off task
   ├─ GET /            live.html  (polls /recent, /status, /predict)
   ├─ GET /runner      session-runner.html
   └─ POST /runner-log  at the end of a session: save the runner's timestamps into the LIVE folder,
                        run segment.py on it -> labelled.csv, segments.csv, check.png

train.py  reads every sessions/*/labelled.csv -> compares models -> REPORT.md + WristStream/model.pkl
receiver.py notices model.pkl changed and reloads it within a second, no restart.
```

The watch does no classification. It does not know whether a session is "training" or "demo"; the same START button serves both.

---

## 5. The watch app (`FocusWatch/`)

**Device.** Samsung Galaxy Watch5 Pro (model SM-R920), which turned out to run Android 16 / Wear OS 6. The app has minSdk 30 and targetSdk 34, so any Wear OS 3+ watch should work. It was stress-tested on Wear OS 5 (API 34) and Wear OS 6 (API 36) emulators before the real watch was available.

**What it records.** At 50 Hz, driven by accelerometer events, taking the most recent value from each other sensor. 16 columns:

| Column(s) | Unit | Source |
|---|---|---|
| `time` | epoch seconds in the POST body; the receiver writes it to CSV as epoch **nanoseconds** | system clock offset + sensor event time |
| `userAccelerationX/Y/Z` | g | `TYPE_LINEAR_ACCELERATION` (gravity removed by the watch) |
| `gravityX/Y/Z` | g | `TYPE_GRAVITY` (which way is down, in the watch's frame = wrist tilt) |
| `rotationRateX/Y/Z` | rad/s | `TYPE_GYROSCOPE` |
| `roll, pitch, yaw` | rad | from `TYPE_ROTATION_VECTOR` |
| `accX/Y/Z` | g | `TYPE_ACCELEROMETER` (raw, gravity included; clips at ±8 g) |

The first 13 names match Apple's Core Motion names so one labelling script reads both. One real difference: **Android's gravity vector points up, Apple's points down.** It only matters if Apple and Galaxy data are ever mixed. They are not.

`userAcceleration` and `gravity` are computed on the watch by Android's sensor fusion. `accX/Y/Z` is the only fully raw signal, and it is what an accelerometer-only study would have had.

**How it survives the screen turning off** (this was the main risk, and why it is a native app):

| Risk | What the app does |
|---|---|
| App paused when the screen sleeps | Foreground service of type `health`, permanent notification, partial wake lock |
| Doze / battery saver | `setup-watch.sh` adds the app to the battery-optimisation whitelist and allows background running |
| Watch drops Wi-Fi or routes through a phone | The app requests a Wi-Fi network explicitly for as long as it records, then falls back to any network |
| Laptop unreachable for a while | Samples queue on the watch (up to 30 min) with sequence numbers and are sent when it returns |
| Laptop down for good, or a crash | Every sample is also written to a CSV on the watch; `pull-backup.sh` retrieves them |
| App killed | The service is sticky: Android restarts it and it resumes with a new session id |

**Emulator test results (12 Nov):** 751 samples in a 15 s test with every interval exactly 20 ms; 50 Hz held with the screen off and the device forced into deep Doze; the receiver killed for 30 s with no gap afterwards; process killed with `kill -9` and recording resumed in about 1 s; a 100 s session delivered 4991 of 4996 samples (largest gap 0.13 s). **Real watch:** every recording in `sessions/` shows 49.8–50.5 Hz with no gap over 0.1 s.

**Setting up the watch** is in `FocusWatch/WATCH-SETUP.md`. In short: enable Developer options on the watch, turn on ADB debugging and Wireless debugging, run `bash setup-watch.sh`, type the pairing address and code, then the connect address. It installs, exempts the app from battery saving, starts the receiver, points the watch at the laptop, and runs a 15 s test that prints PASS or FAIL. Those steps were checked against Google's "Debug Wear OS over Wi-Fi" page and Samsung's "Connect Galaxy Watch to Android Studio over Wi-Fi" guide.

Notes:
- **The watch stores the laptop's address and does not follow it to a new network.** On a new Wi-Fi, open the app, tap the address under the button, type the laptop's current IP (the app adds `http://` and `:8000`). The address can only be edited while not recording.
- The watch and laptop must be on the **same Wi-Fi**, and that network must let devices talk to each other. Campus Wi-Fi often does not; a phone hotspot is the fallback. This has **not** been tried on campus Wi-Fi and is a risk for the demo.
- The scripts assume macOS, `adb` on the PATH or in `~/Library/Android/sdk/platform-tools`, and bash. They have never been run on Windows or Linux.
- `build.sh` needs JDK 17 and the Android SDK (platform 35, build-tools 35). It writes build output to `~/Library/Caches/FocusWatch` (macOS path; change `build.gradle.kts` on another OS). You only need to build if you change the app; the APK is included. The default laptop address compiled into the app is a placeholder (`192.168.0.10`); `setup-watch.sh` overwrites it.
- **Server mode** (streaming to a public server instead of the laptop) needs no app change, since the watch posts to whatever address is in its field, `https://` included. Steps are in `FocusWatch/README.md`. Never needed so far, never tested end to end.

---

## 6. The laptop side (`WristStream/`)

`receiver.py` uses only the standard library plus numpy, and imports `live_model.py`.

```bash
cd WristStream && python3 receiver.py            # port 8000, saves to ../Data Collection/sessions
python3 receiver.py 8010 /some/folder            # another port, another output folder
```

| Endpoint | What it does |
|---|---|
| `POST /ingest` | Body `{"session", "cols", "rows", "device", "os", "hz"}`. Appends rows to `LIVE_<session>/stream.csv`; writes `meta.json` on the first batch |
| `GET /` | `live.html`, read from disk on every request, so page edits need no restart |
| `GET /status` | connected?, rate, sample count, delay |
| `GET /recent?since=<epoch s>` | plotted columns of rows newer than `since` (for the graphs) |
| `GET /predict` | current label, probabilities, on/off task, model name, accuracy, nudge state, 2-minute history |
| `GET /runner` | the session runner page |
| `POST /runner-log?name=<file>` | saves the runner's timestamp CSV into the folder of the recording in progress, waits 3 s for the last samples, runs `segment.py`, returns its output. A second runner file for the same recording is **merged** (block numbers shifted), and the earlier file is kept as `*.partN` |
| `GET /check?folder=LIVE_...` | that session's `check.png` |

**The live page** shows three graphs over the last 10 s (motion, rotation, wrist angle), drawn about 1.3 s behind real time so that once-a-second batches scroll smoothly; the predicted activity; ON/OFF TASK; the probability of each class; a 2-minute on/off strip; the model's name; and a red banner after `NUDGE_S = 10` seconds off task in a row. It deliberately separates **confidence** (how sure the model is about this window) from **accuracy** (measured once on a test session). With no test session the accuracy line reads "not measured yet".

**The model contract** (`live_model.py`): `load()` returns an object with `.name`, `.window_s`, `.test_accuracy`, `.accuracy_kind` and `.predict(win) -> (label, {label: probability})`, where `win` maps column names to numpy arrays covering the last `window_s` seconds. If `model.pkl` is missing, a **placeholder** with hand-set thresholds runs and the page labels it in orange as not a result. A model that throws shows "model error" on the page and never interrupts recording. Set the environment variable `FOCUS_MODEL=/path/to/other.pkl` to test a model without replacing the deployed one.

---

## 7. Recording and labelling

### The method

One continuous watch recording per session. The runner page shows the current activity and a countdown, turns the whole screen red when the time is up, and logs a timestamp each time the person clicks to move on. Before every click the person **claps three times**. Two reasons:

1. The claps are visible in the signal, so the script can check the laptop clock against the watch clock and correct it.
2. Each label ends one second before the closing clap, so no clap motion is ever labelled as an activity.

### The runner (`http://localhost:8000/runner`)

Opened from the receiver (not as a file), it shows whether the watch is streaming, refuses to start silently without it, and files and labels the session at the end with nothing to move by hand.

Settings: session name, participant, watch wrist, dominant hand, posture, a free-text "what's different this session", seconds per block (default 60), rounds (default 2), and which activities to include. The block order is shuffled with a seed derived from the session name, a fresh shuffle per round, never the same activity twice in a row.

Buttons: the big one (or Space) moves to the next block; **Redo this one** throws the current block away and restarts it; **Add a note** attaches text to the current block; **Stop the whole session** ends everything (it was once pressed by mistake when it was called "End session early", which is why one recording holds two runs).

Output: `<SESSION>_<PARTICIPANT>_runner.csv` with columns `session, participant, watch_wrist, dominant_hand, posture, index, round, activity, event, epoch_ms, iso_time, note`. Events: `session_open`, `start`, `end`, `discard`, `note`, `session_end`.

### Activities

| Key | What the person does | Counts as |
|---|---|---|
| `typing` | types on the laptop | on task |
| `writing` | handwrites on paper | on task |
| `idle` | reads, hands resting | on task |
| `drinking` | sips, puts the cup down, repeats | on task |
| `hold_cup` | holds a cup without drinking | on task |
| `calculator` | uses a calculator | on task |
| `other_desk` | stretches, scratches, reaches, fidgets | on task |
| `phone_scroll` | scrolls a feed, phone in the **watch** hand | off task |
| `phone_text` | types a message, phone in the watch hand | off task |
| `phone_call` | phone at the ear with the watch hand | off task |

The on/off split lives in one place: `ON_TASK` and `OFF_TASK` in `Model/features.py`.

### Session names decide the role

`train.py` reads the session name from the runner file:

| Name | Role | Format |
|---|---|---|
| `S1` … `S6` | training | all activities × rounds × seconds per block |
| `TEST` | final score only, never used to choose anything | same format |
| `NATURAL` | extra test | unscripted: real work 3 min, real phone use 1.5 min, three times (`natural_work`, `natural_phone`) |
| `OTHERHAND` | extra test | the three phone activities with the phone in the hand **without** the watch (`phone_*_other`) |
| `PILOT` | ignored by training | 2 blocks × 20 s, to check the setup |

**Recording order in the design:** PILOT → S1 → S2 → TEST → S3 → NATURAL → OTHERHAND. Between sessions: press STOP on the watch, take it off and put it back on, press START, and change something small (seat, cup, where the phone lies). A full session is 20 blocks, about 24 minutes. S1 + S2 + TEST is the minimum for an honest result. Several short blocks are better than one long one: each block is a fresh start with the hand placed slightly differently, which is the variation the model has to cope with.

### What `segment.py` does

```bash
cd "Data Collection" && python3 segment.py sessions/LIVE_20261112-233400
```

1. Reads the recording (`stream.csv`; it also understands Sensor Logger exports and the HemiPhysioData app's files, see section 11).
2. Computes dynamic acceleration and finds **clap bursts**: 3 or more peaks above 1.5 g within 1.5 s (`--threshold` changes the 1.5 g).
3. Takes the block starts and ends from the runner clicks, dropping redone takes.
4. **Clock check:** searches shifts of ±120 s for the one where the most clicks have a clap ending 0.3–6 s before them, and applies it if it beats zero shift.
5. Matches each click to the last clap that came after the previous click and at most 15 s before this one.
6. Labels each block from 3 s after its start click to 1 s before its closing clap. If no clap is found, or the block's end is not immediately followed by another block, the end is the click minus 6 s. Blocks left with under 5 s are skipped.

Outputs in the session folder:
- `labelled.csv`: one row per sample. `t` (s from the start), `epoch_s`, standardised `acc_*`, `gyro_*`, `grav_*`, then **every original column unchanged**, then `label` (`none` outside blocks), `block` (the runner's block index, −1 outside), `session`, `watch_wrist`, `participant`.
- `segments.csv`: one row per labelled block, with start, end, duration and whether the end came from a clap.
- `check.png`: the signal with claps (orange), clicks (blue dotted) and labelled spans. Each orange bar should sit just left of a blue line.

`stream.csv` is never modified by anything.

---

## 8. The recordings in `Data Collection/sessions/`

All from the Galaxy Watch5 Pro on the **left wrist** of a **right-handed** person, sitting, 12 Nov 2026. All are 49.8–50.5 Hz with no gap over 0.1 s. Folder names are the watch's local start time.

| Folder | Length | Labelled? | What it is |
|---|---|---|---|
| `LIVE_20261112-080550` | 13 s | no | the setup script's 15 s connection test |
| `LIVE_20261112-081101` | 5.5 min | no | first real recording, a free mix of activities. `pilot-unlabelled-check.png` shows k-means finding four clear clusters in it without labels |
| `LIVE_20261112-082701` | 10 s | no | short test |
| `LIVE_20261112-083255` | 2.9 min | no | recorded without the runner; contents unknown |
| `LIVE_20261112-130029` | 2.6 min | no | recorded without the runner; contents unknown |
| `LIVE_20261112-233008` | 13 s | no | connection test on a new network |
| **`LIVE_20261112-233400`** | **12.1 min** | **yes** | **the only labelled session (S1)**: 8 blocks. Read its `NOTES.md` |
| `LIVE_20261112-235320` | 4.2 min | no | trying the trained model live; no record of what was done when |
| `LIVE_20261112-235824` | 1.3 min | no | same |

**About the labelled session.** It holds two runs of the runner in one watch recording: the first ended by an accidental click after `hold_cup` and `drinking`; the second stopped after 6 of 20 blocks (`writing, phone_scroll, other_desk, phone_text, typing, phone_call`). The two timestamp files were merged (`S1_P1_runner.csv`; originals kept as `.part1` and `.part2`). One edit was made from the recorder's own note: `hold_cup` was cut at 72 s because the bottle was put down then. Labelled durations run from 48 s (`drinking`) to 86 s (`typing`). **There is no `idle` and no `calculator`.** The recorder's notes, which matter for interpreting it:
- `hold_cup` and `drinking` used a water bottle, not a cup.
- `writing` was done with the **left, non-dominant hand** so that the watch would see it, with some putting the pen down.
- `typing` included pauses with the fingers resting on the keys.
- `other_desk` was scratching, a little stretching, fidgeting with the pen.

The unlabelled recordings are raw material only: usable for looking at signals, for unsupervised exploration, or as replay input for the live page. Without labels they can't be used for training or testing.

---

## 9. The modelling pipeline (`Model/`)

### What `train.py` does

1. **Loads** every folder in `sessions/` that has a `labelled.csv` (and labels any that have a runner file but no labels yet). Folders starting with `_` are skipped. Sessions missing the Galaxy column names are skipped with a message.
2. **Windows.** For each labelled block, half-overlapping windows of 1, 2 or 4 s. Windows never cross a block boundary or a time jump over 0.5 s.
3. **Features** (`features.py`), per window:
   - *time:* for each axis and for the magnitude of each sensor: mean, standard deviation, min, max, interquartile range, mean absolute sample-to-sample change; plus the three between-axis correlations per sensor.
   - *frequency (optional):* share of spectral energy in 0–1, 1–3, 3–8 and 8–25 Hz, dominant frequency, spectral entropy, log energy, per axis and per magnitude.
4. **Compares 210 combinations**: 3 window lengths × 5 sensor sets × 2 feature sets × 7 models.

   | Sensor set | Columns | Why it is there |
   |---|---|---|
   | `acc` | `accX/Y/Z` | raw accelerometer only: the reference paper's setting, and what the cheapest tracker has |
   | `gyro` | `rotationRateX/Y/Z` | does rotation alone carry it? |
   | `acc+gyro` | both | the usual IMU pair |
   | `motion+angle` | `userAcceleration*` + `gravity*` | the watch's own split of the accelerometer into movement and wrist tilt |
   | `all` | movement + tilt + gyro | everything |

   Models: Gaussian Naive Bayes, logistic regression, k-nearest neighbours (k=7), SVM with an RBF kernel, random forest (300 trees), histogram gradient boosting, a small neural net (64-32). Scale-sensitive ones sit behind a `StandardScaler`.
5. **Scores each combination by leave-one-session-out**: train on all training sessions but one, score on the one left out, rotate, average. The metric used to choose is **macro-F1** over activities; accuracy and on/off-task accuracy are recorded too.
6. **Chooses** the best macro-F1. Then chooses the live smoothing length (average of the last 1, 3 or 5 windows' probabilities) by on/off accuracy, again on training sessions only.
7. **Retrains** the chosen combination on all training sessions and scores it **once** on TEST: accuracy, macro-F1, on/off accuracy, a confusion chart, per-activity recall and precision, the five most common mistakes. It also scores one fixed baseline on TEST: accelerometer only, time features, Naive Bayes, standing in for the reference paper's setting.
8. **Extra tests** if present: NATURAL (on/off accuracy, false-alarm share, share of phone use caught, a timeline) and OTHERHAND (share recognised as off task, and what the rest was mistaken for).
9. **Writes** `results/<time>/report.md`, `all_combinations.csv` and figures; copies the report to `Model/REPORT.md`; saves `WristStream/model.pkl`.

### Why sessions are held out and not random windows

Neighbouring windows overlap by half and look alike. A random split puts near-copies on both sides and reports scores that collapse on a new day. Holding out whole sessions is the honest test of "does it work the next time I put the watch on". It helps to say, next to any number, which kind of split produced it. Evaluation and explanation carry 25% of the grade.

### Fallbacks, and what they do to the numbers

- **Only one training session** (today's situation): there is no session to leave out, so it holds out a quarter of every block, in time order, four ways. The report says so at the top. These scores are **optimistic**.
- **No TEST session but three or more training sessions:** the last training session is set aside as the test set, and the report says so.
- **No TEST and fewer than three training sessions:** no final score; the report shows the cross-validation result only and the live page says "not measured yet".

### What is in `model.pkl`

A pickled dict: `model` (the fitted scikit-learn estimator or pipeline), `config` (`{"window_s", "sensors", "freq"}`), `smooth_n`, `name`, `test_accuracy` (on/off-task accuracy on TEST, or `None`), `test_accuracy_kind`, `test_activity_accuracy`, `classes`. Pickles are tied to the scikit-learn version that wrote them (`requirements.txt` pins it); if loading fails, rerun `train.py`. Only load pickles you trust.

---

## 10. Current results and what they mean

From `Model/REPORT.md` (run of 12 Nov 23:51, 47 s, on the one labelled session):

- **Chosen:** random forest, `motion+angle`, 2 s windows, time + frequency features (110 numbers per window), no smoothing. 523 training windows.
- **Held-out-quarter scores:** macro-F1 0.767, activity accuracy 76.5%, on/off-task accuracy 98.3%. Per fold: 0.705, 0.837, 0.850, 0.677.
- **Sensor choice barely mattered** on this data: best macro-F1 per sensor set was 0.767 (`motion+angle`), 0.762 (`acc+gyro`), 0.760 (`acc`), 0.756 (`all`). Raw accelerometer alone was within a point of the best. Do not read much into differences this small on eight minutes of data.
- **Per-activity recall** for the chosen setup (same held-out quarters):

  | Activity | Recall | Mostly confused with |
  |---|---|---|
  | phone_call | 0.98 | — |
  | hold_cup | 0.95 | — |
  | phone_text | 0.91 | phone_scroll |
  | drinking | 0.89 | other_desk |
  | phone_scroll | 0.88 | phone_text |
  | typing | 0.79 | writing |
  | other_desk | 0.48 | drinking, typing |
  | writing | 0.43 | typing, other_desk |

**How to read this.**
- The phone activities separate well from everything else, and mostly get confused with each other, which costs nothing for the nudge. That is the encouraging part, and it matches the unlabelled pilot, where wrist tilt alone split the recording into clean clusters.
- Writing, typing and "other" blur together. All three are on task, so this hurts the activity score but not the on/off decision.
- **None of this has been tested on a different sitting.** In the live trial right after training, holding the phone was mistaken for drinking and typing for writing, among others. Likely contributors: the model has never seen `idle` or `calculator` and must force stillness into some other class; it saw each activity for about a minute in one hand position; and live use includes transitions between activities, which were never in the training data.
- The live model currently has **8 classes, no `idle`**. Sitting still will be reported as something else.

---

## 11. How it got here, and why each decision was made

What was tried, in order.

**The Apple Watch route failed.** The original plan was an Apple Watch Series 3 (watchOS 8.8).
- *Sensor Logger* (App Store) needs watchOS 9.4. Not installable.
- *HemiPhysioData* (App Store, watchOS 6.1+) records to a CSV and worked for a 30 s test at 100 Hz, but a longer recording crashed and saved nothing, and it cannot stream live, which the demo needs. `segment.py` can still read its files; the file `u01_misc_...csv` is that test.
- *Our own app* (`WristStream/WristStream/`, Swift, HealthKit workout session to stay alive, HTTP POST) compiles, but could never be installed. Current Xcode no longer supports watchOS 8 or the Series 3's 32-bit chip; an older Xcode builds it but cannot reach the watch, because once the paired iPhone is on iOS 17 or later the connection uses a protocol watchOS 8 does not speak (Apple Developer Forums thread 742243 describes the same dead end). This route was dropped; the source is kept as a record.

**So: a borrowed Galaxy Watch5 Pro and a native Wear OS app.** Native, not a web page or a phone relay, because the app must keep sampling with the screen off and must not depend on a phone. Direct HTTP to the laptop because the demo needs a live stream and the team wanted the phone out of the loop.

**Which hand holds the phone.** The watch only feels the wrist it is on. A phone in the watch hand changes that wrist's angle and motion, which is learnable. A phone in the other hand leaves the watch wrist resting, which looks like idle. Decision: train with the **phone in the watch hand**; record OTHERHAND as a test-only session to *measure* the blind spot for the report; do not train on other-hand data, which would mostly teach the model "idle".

**Demo wrist.** The model learns the left wrist. On the right wrist the watch is mirrored, so axes flip and gravity points elsewhere. Demo on the left wrist and say why. An untested idea: also train on mirrored copies of the data (flip one axis) so one model covers both wrists.

**Clap-based labelling** was chosen over filming or a second device because it needs nothing but the laptop and survives clock drift between watch and laptop.

**`other_desk` was added** because real use contains movement that is neither work nor phone, and without a class for it the model must force it into one.

**Many short blocks, not long ones,** for the reason given in section 7.

**The reference paper.** Mosin, Min & Salamy, "Using a Wrist-Worn Accelerometer to Detect Off-Task Activities While Studying or Working", CSCI 2024 proceedings, Springer CCIS vol. 2509 (2025), doi 10.1007/978-3-031-94953-1_12. Only the abstract could be read (the full text is behind Springer's login): a MetaBase wrist accelerometer, five desk activities, several features and algorithms compared, Naive Bayes best at up to 81%. Sampling rate, window length, features, number of participants and the train/test split are **not known**. The PDF should be reachable through the university library; the report will need to cite it accurately, and the comparison in `train.py` ("accelerometer only, Naive Bayes") is only a stand-in for their method.

**A public dataset as a second phase** was considered: WISDM Smartphone and Smartwatch Activity and Biometrics (UCI; 51 people, 18 activities, 20 Hz, watch on the dominant hand). It includes typing and, as far as could be told, writing and drinking, but **no phone use**, so it cannot teach off-task. The project guide says new training and evaluation use student-collected data and allows published resources as credited baselines. Not started. Expect it to transfer badly (different watch, wrist, rate, people); that would itself be a reportable result.

---

## 12. Open questions and known problems

Ordered by how much they matter.

1. **Not enough data.** One short session, 8 of 10 activities, no TEST. Everything else is secondary.
2. **Which hand writes?** The watch is on the left wrist of a right-handed person. Natural handwriting is done by the hand *without* the watch; in the recorded session the person wrote left-handed so the watch would see it, which is a movement they would never make in real use. One option: do every on-task activity naturally. The watch will then see handwriting as a resting wrist, indistinguishable from reading, and both are on task, so the nudge is unaffected. In that case the existing `writing` block would be dropped or relabelled, and the report would note that some on-task activities are not separable from a single wrist.
3. **Is idle on task or off task?** Here it is on task (the person could be reading or thinking). The reference paper counted sitting idle as distracted. The team has not decided. It is a one-line change in `features.py`, but it changes what the nudge fires on. Either way the report can state the choice and how it compares with the paper.
4. **Transitions.** Training data contains only clean blocks. Live use is full of switching between activities. NATURAL sessions are the planned way to measure this; none recorded yet.
5. **Other people.** One participant. The professor will probably try it, and accuracy on a new person will be lower. Either record a short session from each teammate (S4, S5) to measure it, or state the limitation.
6. **Campus Wi-Fi** may block the watch from reaching the laptop. Untested. Try it before the demo; have a phone hotspot ready.
7. **Model choice by activity macro-F1, not by on/off accuracy.** The demo depends on on/off. A combination that is slightly worse at naming activities but better at phone-vs-work would currently lose. Worth reconsidering once there is real data; `all_combinations.csv` has both numbers for every combination.
8. **Smoothing.** On the single session, no smoothing won (98.3% vs 95.6% for 3 windows), probably because blocks are short and smoothing smears block edges. On longer natural use, smoothing should help stability. Re-examine with NATURAL data.
9. **Class imbalance** is not handled, because the protocol produces balanced data. Merged or partial sessions (like the current one) are not balanced.
10. **The watch app's clock** is the watch's system clock. It was within about a second of the laptop's in every session, and `segment.py` corrects up to ±120 s using the claps.
11. `receiver.py` files a runner log under the most recent watch session it has seen. If the watch restarted mid-session (new session id), the runner's clicks span two folders and only the second gets labelled. Has not happened; would need a manual merge.

---

## 13. Things noticed along the way

- A random train/test split over windows inflates every score, because neighbouring windows overlap. Sessions are used as the unit.
- `train.py` never uses TEST to choose a model, feature set, window or smoothing length. That is what keeps the TEST score meaningful.
- The held-out-quarter numbers from a single session are a sanity check, not an accuracy figure.
- `features.py` is shared by training and the live page. After changing it, an old `model.pkl` no longer matches and needs retraining.
- Time units differ by file: `stream.csv` has epoch nanoseconds; `labelled.csv` has `t` in seconds from the start and `epoch_s`; the runner has `epoch_ms`.
- `FocusWatch/local.properties` holds a machine-specific SDK path and is in `.gitignore`.
- Apple and Galaxy recordings have opposite gravity signs.
- If `model.pkl` is absent, the live page runs hand-set placeholder thresholds and says so in orange.

---

## 14. What the course requires

From the final-project guide (the PDF every team member has):

- **One email** to the instructor, subject **`CMPS297AH_FP_Final`**, with a working link to a shared folder containing:
  1. the project report (PDF);
  2. code, dependencies, and a README with setup, hardware and how to run;
  3. the data and reference labels used for evaluation, plus saved model files;
  4. the presentation slides.
- Due **Tue 17 Nov 2026, 23:55**. A demo follows in the last week of classes.
- The guide asks to "keep complete trials or sessions separate", which is what the session-level split does. Data collection and quality carry 15% of the grade; evaluation and explanation 25%. Keep the `check.png` files and the recorder's notes as evidence of data quality.
- The guide does not mention git. A repository link can serve as the shared folder if the instructor can open it; check the link while signed out.
- The final exam is a short oral that includes questions about the project.

---

## 15. Environment

Built and tested on macOS (Apple Silicon), Python 3.11, with the package versions in `requirements.txt`. The watch app: JDK 17, Gradle 8.11.1 (wrapper included), Android Gradle Plugin 8.7.3, Kotlin 2.0.21, compileSdk 35, minSdk 30, targetSdk 34. Nothing here has been run on Windows or Linux. The Python parts use nothing platform-specific; the shell scripts and the build output path are macOS-flavoured.

## 16. Possible next steps

- Record full sessions: S1, S2 and TEST first, then S3, NATURAL and OTHERHAND. Questions 2 and 3 in section 12 affect how the sessions are performed and labelled.
- Retrain (`cd Model && python3 train.py`) and read `REPORT.md`.
- Try the model live again. The confusion chart and the NATURAL timeline show where it goes wrong.
- Ideas not started: choosing the model by on/off accuracy, merging on-task classes that can't be told apart, mirrored-wrist augmentation, a class for transitions, the WISDM comparison.
