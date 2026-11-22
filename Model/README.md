# Focus Police: models

Trains on the sessions in `../Data Collection/sessions/`, compares a short list of setups, reports, and hands the best model to the live page. Version 2 (22 Nov 2026); the experiments behind it are in [`lab/FINDINGS.md`](lab/FINDINGS.md).

| File | |
|---|---|
| `train.py` | load, compare, pick, test once, report, deploy |
| `features.py` | signal → numbers, shared with the live page (`../WristStream/live_model.py`) so live and training can't drift apart. Holds calibration, the version 2 features, and which activities count as on task |
| `REPORT.md` | the latest report (written by `train.py`) |
| `results/<time>/` | every run: `report.md`, `all_combinations.csv`, figures (timeline of the test session, confusion chart) |
| `lab/` | the experiments: what was tried, what helped, what didn't. Re-run when sessions are added |

## Run it

```bash
cd Model && python train.py
```

About 2 minutes for three sessions. `--quick` is a sanity run; `--no-deploy` leaves the live page's model alone.

## How it works

- **Each session is one continuous stream**, predicted every 0.5 s like the live page, transitions included. Training uses windows that lie inside one labelled activity; the nudge simulation runs over everything.
- **Calibration.** Every session is rotated so its calibration pose (hands resting on the keyboard) points the watch face up. The pose comes from the recording's `calibration.json` (written by the runner or the live page's Calibrate button); older sessions use their first 5 s of typing.
- **Candidates (12):** 2 s or 4 s windows × {motion features only, motion + wrist angle + 10/30 s context, calibrated} × {extra trees, random forest, logistic regression}. The list is short on purpose: with a handful of sessions, comparing hundreds of setups mostly finds noise.
- **Whole sessions are held out, never random windows.** Each candidate is trained on all training sessions but one and scored on the one left out, rotating.
- **Choice:** best balanced phone-vs-work accuracy, then fewest false alarms, then activity macro-F1. Then the live smoothing (average of the last 1, 5 or 9 predictions) by fewest simulated false nudges.
- **TEST is used once**, after the choice. The report puts it next to an ablation (what wrist angle/context and calibration each add) and the reference paper's setting (raw accelerometer, Naive Bayes).

## Session roles

| Name in the runner | Role |
|---|---|
| `S1` … `S6` | training |
| `TEST` | final score only (if missing and there are 3+ training sessions, the last one is set aside) |
| `NATURAL` | extra test: unscripted work and phone use |
| `OTHERHAND` | extra test: phone in the hand without the watch |
| `PILOT` | ignored |
