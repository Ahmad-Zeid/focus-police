# Focus Police

A watch on the wrist streams motion to a laptop, which works out what you are doing at your desk and nudges you after 10 seconds on your phone. CMPS 297AH Wearable Sensing final project, due Tue 17 Nov 2026, 23:55.

[`HANDOFF.md`](HANDOFF.md) has the full account: what was built, how it was tested, and why each decision was made.

## Where things stand (22 Nov)

- **Detection version 2** Each session now starts with a 5 s **calibration** (hands resting on the keyboard), and the model also reads **wrist angle** and the **last 10–30 s**. On a session from a person the model never saw, phone-vs-work accuracy went from about 68% to 89–94%; on the current test session it is 97% with no false nudges. Measured in [`Model/lab/FINDINGS.md`](Model/lab/FINDINGS.md).
- **Data (22 Nov, evening):** labelled sessions from 4 people (codes, no names): P1 (2), P-8BZH (3), P-MQME (1) for training, and **P2 kept aside as the test person** (2 scripted sessions; their NATURAL session is scored separately). Models are compared by **leaving one person out**: every score is on someone the model never saw.
- **Results** ([`Model/REPORT.md`](Model/REPORT.md)): leaving one person out, balanced phone-vs-work **91.4%** (phone caught 83.9%, false alarms 1.1%); P1 96.7%, P-8BZH 96.3%, **P-MQME 66.3%** (holds the phone face-up and low, unlike everyone else: the clearest failure case). On the test person P2: **93.6%**, every phone activity nudged, median delay 9 s, but 12.4% false alarms (mostly holding a cup). P2's unscripted NATURAL session: 96.0% of phone time caught, no false alarms. Baselines on P2: reference paper 85.1%, wrist-angle threshold 77.5%, movement threshold 50.0%.
- The live model is refit on every session, TEST and NATURAL included (`train.py --test-participant P2 --deploy-all`).
- **Baselines from the proposal** (`Model/baseline.py`): a single threshold on wrist movement scores 46–53% (about chance), a threshold on calibrated wrist angle 76–85% (it misses phone calls), the reference paper's setting (raw accelerometer, Naive Bayes) 78–79%. The report also has a look-alike table (cup, bottle, drinking vs phone).
- **Focus page for the phone** at `http://<laptop address>:8000/focus`: current status, a timeline of the whole session, focus share, distractions and the longest focus stretch, as the proposal's output describes. Saved per recording as `focus_summary.json`.
- The watch buzzes after **2 minutes** off task, as in the proposal; the live page nudges after 10 s and shows the watch countdown. `FOCUS_WATCH_NUDGE_S=20` shortens the buzz for a demo.
- The runner asks for the calibration automatically; on the live page, press **Calibrate** after starting the watch.
- **Participant study** Recording other people follows a written protocol with read-aloud consent, random participant codes, names kept off the data (a git-ignored roster), and a withdrawal command. Training holds out one person at a time. Everything is in [`Data Collection/STUDY.md`](Data%20Collection/STUDY.md).
- Runs on macOS and Windows (`FocusWatch/setup-watch.ps1` on Windows). Training, analysis and replay need no watch.

## The five commands that matter

Set up Python once (3.10 or newer):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Train on all labelled sessions and write `Model/REPORT.md`:

```bash
cd Model && python3 train.py
```

Train with one or more participants set aside as the test set (the participant study):

```bash
cd Model && python3 train.py --test-participant P-XXXX
```

Start the laptop receiver (live page at http://localhost:8000, session runner at http://localhost:8000/runner):

```bash
cd WristStream && python3 receiver.py
```

Replay a recording into the live page when there is no watch (start a test receiver first with `python3 receiver.py 8010 /tmp/focus-test`):

```bash
cd WristStream && python3 replay.py "../Data Collection/sessions/LIVE_20261112-233400/stream.csv" http://localhost:8010
```

Set up a watch (steps in `FocusWatch/WATCH-SETUP.md`):

```bash
cd FocusWatch && bash setup-watch.sh
```

## Folders

| Folder | What it is |
|---|---|
| `FocusWatch/` | the Wear OS watch app, its built APK and setup scripts |
| `WristStream/` | the laptop side: receiver, live page, live model. (Named after an abandoned Apple Watch app, whose source is also in there) |
| `Data Collection/` | the session runner, the labelling script, and every recording in `sessions/` |
| `Model/` | features, training, reports |

The four folders find each other by relative path.

`localhost` in the addresses above means "this computer": the pages come from the receiver running on your own laptop, so they work for whoever runs it.
