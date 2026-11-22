# Focus Police: data collection

**Recording other people:** use the participant study (session STUDY in the runner). Protocol, consent, privacy and withdrawal: [STUDY.md](STUDY.md).

Use it to record labelled wrist-motion sessions and turn each one into a labelled CSV. The project overview is in `../README.md`; this file is the protocol and its history, newest first.

| File | What it does |
|---|---|
| `session-runner.html` | Guides you through one activity at a time, times each activity, and saves the labels with the watch recording. |
| `segment.py` | Uses watch timestamps to label the recordings and draws `check.png` for review. Older recordings can still use their clap markers. |
| `sessions/` | One folder per session, e.g. `sessions/S1_P1/`. |

---

## 0. Current recording guide

Open **http://localhost:8000/runner** (use the receiver's port if different). Choose a session and participant ID, tap **START** on the watch, then click **Start session**. The runner shows one activity at a time.

0. **Calibrate (new, 22 Nov).** The runner first asks you to rest both hands on the laptop keyboard and press **Calibrate**, then keep still for 5 s. The wrist pose is saved as `calibration.json` with the recording, and the model reads every movement relative to it. It is the largest accuracy gain measured (`../Model/lab/FINDINGS.md`). If it says the wrist moved, try again.

1. Get ready, then click **Start activity**. Do the activity until the timer ends. No clapping.
2. Click **Next activity** and repeat. If the previous attempt went wrong, use **Redo activity** before moving on.
3. Click **Save session**, wait for the label count, then tap **STOP** on the watch. To stop early, **End session** saves completed activities. Start a fresh watch recording for each new session.

**Training and TEST defaults:** two shuffled rounds of eight activities at 60 seconds each (16 minutes plus setup and transitions). The set follows the proposal: laptop typing, handwriting, quiet reading, drinking, holding a cup or bottle, phone scrolling, phone texting, and a phone call. A physical calculator is optional under **Recording options**. Use the same activity set for comparable training and test sessions.

**Wrist placement:** use the same watch wrist throughout the main training and test recordings, and record both the actual watch wrist and writing hand in the runner. Write naturally with your usual hand. If the watch is on the other wrist, handwriting may look like quiet reading. Keep both labels to measure that confusion, and count both as on task when scoring reminders. Hold the phone in the watch hand for the main comparison. Keep any recordings made with a different wrist placement or phone hand separate so we can see how that change affects the result.

The runner aligns each activity start to the live watch stream. The timer supplies the end. Labelling trims three seconds from both ends to leave out transitions and rejects blocks with more than a second of missing data. Expand **Review saved labels** to see the matching plot and any skipped blocks. Each recording, timestamp log, and label file is saved together under `sessions/LIVE_<time>/`.

| Session | Role | What it is |
|---|---|---|
| PILOT | setup check | 20 s of laptop typing and 20 s of phone texting |
| S1, S2, S3 (up to S6) | **training** | The format above. Between sessions: STOP, take the watch off and on, START, and change something small (seat, cup, where the phone lies). Fill in "What's different this session" |
| TEST | **final score only** | Same format. Never used to choose anything. Record it like a normal session, not more carefully |
| NATURAL | extra test | No script: real work 3 min, real phone use 1.5 min, three times. Closest to the demo |
| OTHERHAND | extra test | Phone in the hand without the watch, 3 × 60 s. Measures the known blind spot |

**Order if time runs out:** PILOT → S1 → S2 → TEST → S3 → NATURAL → OTHERHAND. Two training sessions plus TEST is the minimum for an honest result; every extra training session helps more than longer blocks do.

**Then:** `../Model/train.py` compares window lengths × sensor sets × feature sets × models, picks one without looking at TEST, scores it once on TEST, and writes `../Model/REPORT.md`. See `../Model/README.md`.

The sections below document the earlier clap-based and offline workflows.

## 0b. 12 Nov morning: record on the borrowed Galaxy Watch5 Pro

The recording watch is now a **Galaxy Watch5 Pro** running our own app (`../FocusWatch/`). Follow **`../FocusWatch/WATCH-SETUP.md`**: one setup command, then START/STOP on the watch per session.
- Sessions land in `sessions/LIVE_<time>/stream.csv`. Drop the runner's timestamp file into the same folder and run `segment.py` on it, as below.
- An Apple Watch Series 3 was tried first and dropped: our own app could not be installed on it (paired iPhone on iOS 26), and the HemiPhysioData app could not stream live. `segment.py` can still read HemiPhysioData files (section 1).

## 1. How the raw data gets off the watch — UPDATE 11 Nov 14:30: use HemiPhysioData

Sensor Logger needs watchOS 9.4 and won't install on the Series 3 (watchOS 8.8). The app in use is **HemiPhysioData** (free, watchOS 6.1+):
- Settings: Activity **MISC → MISC 1** every time (it's only a tag), **Interval Hertz 60**, longest duration.
- It saves one CSV per recording on the watch; send it to the iPhone, then AirDrop it into the session folder. Several files in one session folder are fine: `segment.py` joins them by clock time.
- First test (13:31, 30 s, typing): steady 100 Hz, no gaps while typing with the wrist down, typing clearly visible (see `pilot-1331-typing-check.png`).
- **Known problem:** a longer test closed itself after about a minute and **saved nothing**. Until a 3-minute test survives, record **one file per activity** (stop and start on the watch at each switch), so a crash loses at most one activity.

The original Sensor Logger instructions below stay for teammates with a newer watch.

**No developer account, no app of our own, no watch interface.** Use an App Store app that records the watch's motion sensors (Core Motion) to a file:

1. On the iPhone, install **Sensor Logger** from the App Store. Then open the **Watch** app on the iPhone and scroll to *Available Apps* to install it on the watch.
   - The Series 3 tops out at watchOS 8. One App Store listing for Sensor Logger needs watchOS 8.4, a newer one needs 9.4. If the App Store offers the **last compatible version**, accept.
2. In Sensor Logger, turn on the watch's wrist motion (accelerometer + gyroscope) at the highest rate it offers (50–100 Hz). Turn everything else off.
3. Start the recording **on the watch**. When you stop it, the recording transfers to the iPhone. Export it from there (recording → share → **Zipped CSV**) and AirDrop it to the Mac.

**If it won't install on the Series 3:**
- **Plan B:** a teammate's newer watch. Same app, same steps, still no account.
- **Plan C, only if both fail:** a minimal recorder app built in Xcode. A **free** Apple ID is enough to put it on your own watch (the app expires after 7 days, which is fine). The paid developer account isn't needed. Cost: about 2–3 h.

## 2. The runner (`session-runner.html`)

1. Fill in the session, participant, watch wrist and dominant hand. Untick any activity you can't do. Press *Open the session*.
2. Start the watch recording. Get into the first activity, **clap 3×**, then click the big button.
3. Work normally. The screen shows the activity, what to do and the time left. When the target is reached the whole screen turns **red**, counts the overtime, and the tab title flashes *SWITCH NOW*. Going over is fine.
4. To switch: stop, **clap 3×**, then click the big button (or press **Space** with that window focused). The click ends the current activity and starts the next one.
5. If something went wrong in an activity (someone talked to you, you used the wrong hand), click **Redo this one**. That take is thrown away; clap 3× and click to start it again.
6. After the last activity: clap 3×, click *finish*, then stop the watch recording. The timestamp file downloads as `<SESSION>_<PARTICIPANT>_runner.csv`.

**Don't stop the watch recording between activities.** One continuous recording per session; the claps and clicks do the cutting.

The claps matter: the clicks give the labels, the claps let the script line the watch clock up with the laptop clock and keep the clap motion out of the labels.

## 3. Pilot (10 min). Do this before any real session.

1. Runner on **PILOT** (2 activities × 20 s). Follow section 2.
2. Make `sessions/PILOT_P1/`, unzip the export into it, and move `PILOT_P1_runner.csv` from Downloads into it.
3. Run:

```bash
cd "Data Collection" && python3 segment.py sessions/PILOT_P1
```

4. Open `sessions/PILOT_P1/check.png`. Orange bars are the claps the script found, blue dotted lines are your clicks, and the shaded areas are what got labelled. Each orange bar should sit just left of a blue line.
   - **Claps missing:** clap harder, or rerun with `--threshold 1.0`.
   - **Extra orange bars:** rerun with `--threshold 2.0`.
   - It prints which file it read. If that's the wrong file, add `--file <path>`.

When this section was written the script had only run on synthetic data. It has since run on real Galaxy recordings (see `sessions/`).

## 4. Which hand does what (decided 11 Nov)

The watch only feels the wrist it's on. A phone held in the watch hand changes that wrist's angle and movement, which is learnable. A phone used only by the other hand leaves the watch wrist resting, which looks like idle.

- **Training data: phone in the watch hand.** Scrolling, texting and the call are all done with the phone in the watch hand (texting with both thumbs is fine). The runner says this on every phone activity.
- **One short extra session, OTHERHAND (~8 min):** the three phone activities with the phone in the hand **without** the watch. Never train on it. Use it only to test how badly the model misses when the phone is in the other hand. That's a measured limitation for the report, which is graded under *evaluation and explanation* (25%), instead of a guess.
- **Recording both hands as training classes is out.** It doubles the phone recordings, and the other-hand data would mostly teach the model "idle".

**For the demo:** the model learns *left wrist*. On the right wrist the watch is mirrored, so the axes flip and gravity points a different way. Accuracy would drop. So:
- Demo it yourself first, then hand it to the professor **on the left wrist**, and say why.
- The professor is also an unseen person. The report should say up front that accuracy on new people is expected to be lower.
- Optional, cheap, if there's time: train on mirrored copies of the data too (flip the sign of one axis). That's a standard trick that lets one model handle both wrists. Test it on a short right-wrist recording before claiming it works.

## 5. Real sessions (about 20 min each)

Each session runs all 9 activities, 2 min target each, in an order that changes per session.

| Session | Who | Notes |
|---|---|---|
| PILOT | you | 1 min, checks the setup |
| S1 | you | sitting, watch on the left wrist |
| S2 | you | another time or desk; take the watch **off and back on** first |
| S3 | you | same again. This is the session to keep for testing |
| OTHERHAND | you | phone activities in the non-watch hand. Test only |
| S4, S5 | teammates (optional) | the unseen-people test, watch on the left wrist |

After each session: export, unzip into `sessions/<S>_<P>/`, move the runner file in, run `segment.py`, and look at `check.png` **before** recording the next one.

Time: pilot 10 min + S1–S3 3 × ~25 min + OTHERHAND ~10 min ≈ **1.5–2 h**, which fits around other work since you're typing during it anyway.

## 6. What the recordings feed

- `labelled.csv` in each session folder is the data. It carries the standardised columns (`acc_*`, `gyro_*`, `grav_*`) plus **every original column of the recording, unchanged** (raw accelerometer `accX/Y/Z`, roll/pitch/yaw, ...). `stream.csv` itself is never modified, and the watch keeps its own backup copy. Activities map to **on-task** (typing, writing, idle, drinking, hold_cup, calculator) or **off-task** (the phone ones).
- The guide asks you to "keep complete trials or sessions separate". Train on S1+S2, test on S3. Test on S4/S5 to claim other people, and on OTHERHAND for the other-hand limitation.
- Graded under **Data collection and quality, 15%**. Keep `check.png` and the notes as evidence.

### Optional phase 2: a public dataset

The guide (p. 8) says "new training and evaluation use student-collected data" and allows published resources as **credited baselines**. So a public dataset can be a comparison, but our recordings stay the test set.

- **Best candidate: WISDM Smartphone and Smartwatch Activity and Biometrics (UCI, CC BY 4.0).** 51 people, 18 activities, 3 min each, watch on the **dominant** hand, 20 Hz. It includes typing, and probably writing, drinking from a cup and sitting too; check `activity_key.txt`. Known issues: uneven sampling and gaps (reported by a later cleanup project).
- **There's no phone use in it.** No public wrist dataset with labelled scrolling, texting or calls turned up. So it can't teach off-task on its own.
- **The test:** train on WISDM's matching classes, test on our S3, then compare with our own model. Expect it to do **worse**, not better: a different watch, a different wrist, 20 Hz, and different people. Even so, that's a real result to report ("more data from the wrong setup doesn't transfer"). The version that might help is mixing WISDM's on-task classes into our training set. Do it only after the main pipeline works.

## 7. Where the git repo comes in

The guide never mentions git. What it asks for (p. 9) is **one email to the instructor, subject `CMPS297AH_FP_Final`, with a working link to a shared folder** containing:
1. the project report (PDF),
2. code + dependencies + a README with setup, hardware and how to run,
3. **the data and reference labels used for evaluation**, plus saved model files,
4. the presentation slides.

The teammate's repo is the natural home for 2 and 3, and it's how all three of you share the recordings. Once the sessions are recorded, this folder (scripts + `sessions/`) goes into the repo, e.g. as `data/`. At 100 Hz a 20-minute session is roughly 10–30 MB of CSV, under GitHub's 100 MB per-file limit. If the instructor can open the repo, its link can be the "shared folder" in the email. Otherwise share a Drive folder containing the PDF, the slides and a link to the repo. Either way, check the link signed out before sending.
