# Focus Police lab: what improves detection, measured

22 Nov 2026. The experiments behind `train.py` version 2. Every number is reproducible with the scripts in this folder.

## The short version

1. **Calibration is by far the largest gain.** Before each session the user rests both hands on the keyboard for 5 s; every movement is then measured relative to that pose. On the session from a *different person*, phone-vs-work accuracy went from **68% to 94–97%**. It removes how the strap sits and each person's resting posture, which is what made the model fail on a new person.
2. **Wrist angle and the last 10–30 s help**, mostly together with calibration. The model's most-used features are now wrist tilt (pitch, gravity direction) over the window and over the last 10–30 s.
3. **Extra trees** is the best of the models tried. Predicting activity names and adding up the phone ones works as well as predicting phone-vs-work directly.
4. **Rejected because they didn't help:** rotation augmentation (made things worse), an HMM smoother, a 1D CNN on raw signals (tied, adds PyTorch), "how long the pose has been held" features, 60 s context, and a running calibration with no user step (much worse).
5. **Simulated nudges:** with these changes, zero false nudges across all three held-out sessions and every phone activity nudged, a median of 9–11 s after it starts.

## How it was measured

- **Data:** the three labelled training sessions: S1 by P1 (12 Nov, 8.9 min labelled) and S1 + S2 by P-8BZH (22 Nov, 13.7 and 14.7 min). Same watch (Galaxy Watch, left wrist), two people.
- **Leave one session out.** Train on two sessions, predict the third, rotate. Holding out **P1's session** is the cross-person test: the model has only seen P-8BZH.
- **Continuous streams.** Every session is predicted every 0.5 s from start to end, including the transitions between activities, exactly like the live page (`data.py`). Training uses only windows inside one labelled activity.
- **Scores** (`evaluate.py`): *balanced on/off* = mean of phone caught and work left alone; *false alarms* = work windows flagged as phone; *false nudges/h* = a nudge fires after 10 s flagged in a row, and it is false if it fires during work.
- **Everything is causal:** a feature at time T uses only samples up to T, so whatever wins here can run live.

## Results

Balanced phone-vs-work accuracy on each held-out session (P1 = the other person):

| Experiment | Mean | P1 | P-8BZH S1 | P-8BZH S2 | False alarms |
|---|---|---|---|---|---|
| Old pipeline features, random forest | 85.8% | **67.7%** | 95.6% | 94.2% | 5.0% |
| + wrist angle | 86.0% | 67.8% | 95.9% | 94.4% | 5.4% |
| + 10/30 s context | 86.7% | 71.8% | 93.4% | 94.8% | 3.8% |
| + wrist angle + context | 86.0% | 69.9% | 93.4% | 94.8% | 4.3% |
| **+ calibration, whole session (offline upper bound)** | 94.7% | 95.1% | 96.8% | 92.1% | 5.1% |
| **+ calibration, 5 s typing pose (what ships)** | 94.0% | **93.8%** | 91.8% | 96.4% | 4.9% |
| + calibration, running estimate, no user step | 76.5% | 86.8% | 84.6% | 58.1% | 17.4% |
| calibrated, 4 s window | 94.8% | 96.1% | 91.2% | 96.9% | 3.7% |
| calibrated, old features only | 93.2% | 94.4% | 93.8% | 91.6% | 4.8% |

Models (calibrated, 4 s window, angle + context):

| Model | Activities → phone | Phone vs work directly |
|---|---|---|
| **Extra trees** | **95.7%** | 95.5% |
| Random forest | 94.8% | 93.8% |
| Gradient boosting | 92.8% | 91.8% |
| Logistic regression | 90.6% | 88.0% |
| 1D CNN on raw calibrated signals (`deep.py`, 3 seeds) | 95.5% | |
| CNN + extra trees averaged | 96.0% (but a false nudge) | |

Things that did not help (extra trees, calibrated):

| Experiment | Balanced | False nudges/h |
|---|---|---|
| no augmentation | **95.7%** | 2.4 |
| 2 rotated copies ±10° | 95.3% | 2.4 |
| 4 rotated copies ±15° | 94.2% | 8.5 |
| 4 rotated copies ±25° | 92.9% | 13.3 |
| + "seconds in the same pose" features | 95.4% | 0 |
| + 60 s context | 94.2% | 0 |

Smoothing (`tune_smoothing.py`): averaging the last 9 predictions (4.5 s) at threshold 0.5 gave **0 false nudges** at 95.5% balanced; the HMM filter was worse at every setting (94.4–95.1%, 4.8–12 false nudges/h). `train.py` picks the averaging length from 1, 5 and 9 by fewest false nudges on held-out sessions.

## Why calibration works

The watch measures the wrist's angle relative to gravity. The same activity looks different when the strap is rotated a few degrees, or when someone rests their forearms higher. `look_calibration.py` shows the sessions' resting poses differ by 34–68°, and that after calibration the same activity lands at similar angles across people (typing pitch: P1 −12° vs P-8BZH −30° before; 16° vs 30–33° after).

The rotation is computed from the median gravity direction of the 5 s pose and turns it to "watch face up" (`features.calibration_rotation`). Every 3-axis signal (movement, gravity, gyroscope) is rotated by it before features are computed. The running estimate fails because its reference keeps moving as the session goes on, so the same pose means different things at different times.

## Which sensors (added 22 Nov, evening)

Asked at the demo: "are you using all three sensors?" The watch records the **accelerometer** and the **gyroscope**, plus Android's fused virtual sensors built from them (linear acceleration = movement, gravity = wrist angle) and a rotation vector. The **magnetometer** is not recorded on its own and no model uses it. Until now every v2 experiment used one sensor set ("all": movement + gravity + gyroscope), so this was tested directly with `sensors.py`: same evaluation as above (extra trees, 4 s, calibrated, leave one session out), only the sensor set changes.

| Sensors | Balanced | False alarms | P1 (other person) | P-8BZH S1 | P-8BZH S2 |
|---|---|---|---|---|---|
| accelerometer only (raw) | 94.5% | 4.9% | 94.6% | 94.8% | 94.0% |
| gyroscope only | 74.0% | 6.2% | 64.7% | 81.9% | 75.3% |
| accelerometer + gyroscope (raw) | 94.4% | 4.6% | 94.1% | 95.5% | 93.5% |
| accelerometer split: movement + gravity | 94.9% | 4.8% | 95.5% | 94.5% | 94.6% |
| all: movement + gravity + gyroscope | 94.5% | 4.8% | 94.7% | 95.2% | 93.7% |
| all + wrist angle + context (shipped) | **95.4%** | **3.4%** | 96.3% | 92.6% | 97.2% |

Reading: the **accelerometer carries the detection**; mostly through gravity, i.e. how the wrist is tilted. The gyroscope alone is much weaker, and adding it to the accelerometer changes nothing measurable (differences of about 1 point are within the noise of three sessions). The shipped model keeps the gyroscope because the stillness and context features use it and it costs nothing on the watch; an accelerometer-only model would do about as well. Same caveats as above: two people, three sessions, simulated calibration.

## What this does not show yet

- **Two people, three sessions.** The cross-person number rests on one held-out session. It is promising, not proven.
- **Some optimism from tuning.** About 30 configurations were compared on these same sessions. The big effects (calibration +25 points on the other person, consistent for every model and window in `train.py`) are far larger than that selection effect; small differences (extra trees vs random forest, ±1 point) are not.
- **The 5 s typing pose was simulated** from the first 5 s of each session's typing activity. New recordings use the real calibration step (the runner asks for it; it is saved as `calibration.json`).
- **Scripted activities.** Real use mixes everything; a NATURAL session will show how transitions behave.
- **Phone calls** are the weakest phone activity (50–95% flagged per session): a call and a sip both bring the hand to the face.

## Next data, in order of value

1. **More people**, each with the calibration step: one or two S sessions per teammate. This is the single biggest lever: same-person scores are about 97%, cross-person about 89–94%.
2. **A fresh TEST session**, ideally from someone not in training. Nothing has looked at it, so it gives the number to put in the report.
3. **NATURAL**: real work and real phone use, switching naturally. Measures transitions and false nudges in realistic use.
4. **More phone calls** (currently 4 blocks in total) and **OTHERHAND** to measure the known blind spot.
5. Keep the activity set the same across sessions (P1's session has `other_desk` but no `idle`, P-8BZH's the opposite).

## Files

| File | What it does |
|---|---|
| `data.py` | loads sessions as continuous streams, cuts windows every 0.5 s |
| `feats.py` | feature families (base, orient, context, dwell, calibration variants) and rotation tools |
| `evaluate.py` | leave-one-session-out, smoothing, nudge simulation, scores |
| `run.py <group>` | runs a group of experiments (features, calibration, models, augment, duration), appends to `results.csv` |
| `tune_smoothing.py` | smoothing and threshold sweep; lists missed phone activities and false nudges |
| `errors.py` | per-activity errors and feature importance |
| `look_calibration.py` | resting pose and activity angles before and after calibration |
| `sensors.py` | which sensors: the same evaluation with each sensor set; writes `sensors_results.csv` |
| `deep.py` | the 1D CNN comparison (needs `pip install torch`; not used live) |

Run from this folder with the project's Python, e.g. `..\..\.venv\Scripts\python.exe run.py calibration`. Features are cached in `.cache/`; delete it after relabelling a session.

## Research context

The published literature has little on detecting phone use from a wrist alone. Related findings used here: wrist accelerometer + gyroscope beat either alone; held-out-subject evaluation gives much lower and more honest scores than random splits; augmentation gains are dataset-dependent (they did not help here); window-level classifiers flicker, and simple averaging of neighbouring predictions gives modest gains while HMMs target transitions. Reference starting point: Mosin, Min & Salamy (2025), doi 10.1007/978-3-031-94953-1_12.
