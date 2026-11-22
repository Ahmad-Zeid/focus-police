# Focus Police

**Telling desk work from phone use with a motion sensor on the wrist.** A smartwatch streams wrist motion to a laptop; a classifier decides every half second whether you are doing desk work or using your phone; after 10 s on the phone the page nudges you, and after 2 minutes the watch vibrates. A phone page shows whether you are focused, the current activity and a timeline of the session.

CMPS 297AH Wearable Sensing, Fall 2026, American University of Beirut. Team muyTracker: Ahmad Zeid, Ali Sleiman, Mohamad Alame.

## What is in this submission

| | Where |
|---|---|
| **Project report (PDF)** | [`Report/Focus Police - Final Report.pdf`](Report/Focus%20Police%20-%20Final%20Report.pdf) (Word version next to it) |
| **Presentation slides (PDF)** | [`Report/Focus Police - Slides.pdf`](Report/Focus%20Police%20-%20Slides.pdf) |
| **Code** | `FocusWatch/` (watch app), `WristStream/` (laptop receiver and pages), `Data Collection/` (session runner, labelling), `Model/` (features, training, evaluation) |
| **Data and reference labels** | `Data Collection/sessions/`: every raw recording (`stream.csv`), its labels (`labelled.csv`, `segments.csv`), the runner's timestamp log and a label check plot (`check.png`) |
| **Saved models** | `WristStream/model.pkl` (main model) and `WristStream/model_instant.pkl` (instant model) |
| **Full evaluation output** | [`Model/REPORT.md`](Model/REPORT.md) and every training run in `Model/results/`; the experiments behind each design choice in [`Model/lab/FINDINGS.md`](Model/lab/FINDINGS.md) |
| **Dependencies** | [`requirements.txt`](requirements.txt) (Python); the watch app builds with Gradle (`FocusWatch/`) |

## Results in brief

All scores are on people the model never saw, predicted every 0.5 s on the continuous recording, as the live page runs. Details and limitations are in the report.

- **Test person (P2), set aside before any choice:** 93.6% balanced phone-vs-work accuracy, 99.5% of phone time caught, every phone activity nudged after a median of 9 s, no false watch vibrations. Most false alarms come from holding a cup, the look-alike the proposal predicted.
- **Unscripted session** (real work and real phone use): 96% of phone time caught, no work flagged as phone.
- **Baselines** on P2: threshold on wrist movement 50% (chance), threshold on wrist angle 77.5% (misses phone calls), the reference paper's setting (raw accelerometer, Naive Bayes) 85.1%.
- **Leaving one person out:** 96.7%, 96.3% and 66.3%. The last person holds the phone differently from everyone else; the report explains this failure.

## Hardware

- A Wear OS 3+ smartwatch. We used two borrowed Samsung watches: a Galaxy Watch5 Pro (SM-R920) and a Galaxy Watch4 (SM-R870), on the left wrist.
- A laptop (macOS or Windows) on the same Wi-Fi as the watch. Campus Wi-Fi may block devices from reaching each other; a phone hotspot works.
- Optional: a phone on the same Wi-Fi to show the focus page.

Training, evaluation and replaying recordings need no watch.

## Set up and run

Python 3.10 or newer, once:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

**Install the watch app** (step by step in [`FocusWatch/WATCH-SETUP.md`](FocusWatch/WATCH-SETUP.md); the built app is `FocusWatch/FocusPolice-watch.apk`):

```bash
cd FocusWatch && bash setup-watch.sh
```

On Windows: `FocusWatch/setup-watch.ps1`.

**Start the laptop receiver.** It prints the address to type into the watch app:

```bash
cd WristStream && python3 receiver.py
```

If port 8000 is taken, pass another one, e.g. `python3 receiver.py 8001`, and type `<laptop address>:8001` on the watch. Then press START on the watch and open:

| Page | Address |
|---|---|
| Focus page (made for a phone): focused or not, the activity, the instant prediction, the session timeline and summary | `http://<laptop address>:8000/focus` |
| Live page: signals, probabilities, nudge | `http://localhost:8000` |
| Session runner: guided, labelled recordings | `http://localhost:8000/runner` |

Before use, calibrate: rest both hands on the keyboard and press **Calibrate** (on the focus page, the live page or in the runner).

**Without a watch,** replay a recording into a second receiver that writes to a scratch folder:

```bash
cd WristStream && python3 receiver.py 8010 /tmp/focus-test
```

```bash
cd WristStream && python3 replay.py "../Data Collection/sessions/LIVE_20261117-120625/stream.csv" http://localhost:8010
```

**Retrain and re-evaluate** (writes `Model/REPORT.md` and `Model/results/<time>/`, then saves both live models):

```bash
cd Model && python3 train.py --test-participant P2 --deploy-all
```

## How it works

| Folder | What it is |
|---|---|
| `FocusWatch/` | Wear OS app (Kotlin): samples accelerometer, gyroscope and gravity at 50 Hz, streams them over Wi-Fi with the screen off, keeps a backup on the watch, and vibrates when the laptop says so |
| `WristStream/` | laptop side: `receiver.py` saves every sample, runs both models every 0.5 s and serves the pages; `live_model.py` loads the models; `focus_log.py` keeps the session timeline; `replay.py` plays a recording back |
| `Data Collection/` | `session-runner.html` guides a recording and logs each activity on the watch's clock; `segment.py` turns a recording into labels; `STUDY.md` is the participant-study protocol; `sessions/` holds the data |
| `Model/` | `features.py` (shared by training and the live page), `train.py` (comparison, evaluation, report, deployment), `baseline.py` (the proposal's threshold baseline), `lab/` (experiments) |

**Main model:** a random forest on a 4 s window plus 10 s and 30 s of context, calibrated to the user's resting pose and trained on mirrored copies so it serves either wrist. Its answer is reported as four classes: desk work, drinking, holding a cup and phone. It decides focused or not and drives the nudge.

**Instant model:** the same classifier on a 2 s window with no context and no averaging, shown under the main prediction, so a single sip or a switch of activity appears within about two seconds.

## Data and privacy

Participants are identified by random codes (P1, P2, P-8BZH, P-MQME). Names and contacts stay in a private roster on the recording laptop, excluded from this repository. Consent, privacy and withdrawal are described in [`Data Collection/STUDY.md`](Data%20Collection/STUDY.md). Sessions without a runner log (connection tests, live tries) are kept as raw data and are not used for training or testing.

## Tools and sources

Python (NumPy, pandas, SciPy, scikit-learn, Matplotlib), Kotlin and Wear OS. Starting point: Mosin, Min and Salamy, *Using a Wrist-Worn Accelerometer to Detect Off-Task Activities While Studying or Working* (2025), doi.org/10.1007/978-3-031-94953-1_12. AI assistants (Claude Code, OpenAI Codex) helped write code; the report lists how.
