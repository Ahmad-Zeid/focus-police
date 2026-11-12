# Focus Police: models

Built 12 Nov 2026. Trains on the sessions in `../Data Collection/sessions/`, compares many setups, reports, and hands the best model to the live page.

| File | |
|---|---|
| `train.py` | the whole experiment: load, window, compare, pick, test once, report, deploy |
| `features.py` | windows → numbers. Shared with the live page (`../WristStream/live_model.py`), so live and training can't drift apart. Also holds which activities count as on task |
| `REPORT.md` | the latest report (written by `train.py`) |
| `results/<time>/` | every run: `report.md`, `all_combinations.csv`, the figures |

## Run it

```bash
cd "Model" && python3 train.py
```

About 4 minutes for four 20-minute sessions on this Mac. `--quick` is a one-minute sanity run; `--no-deploy` leaves the live page's model alone.

## What is compared (210 combinations)

- **Window length:** 1, 2, 4 s (half-overlapping, never across two blocks).
- **Sensors:** `acc` (raw accelerometer only, the reference paper's setting) · `gyro` · `acc+gyro` · `motion+angle` (the watch's split of the accelerometer into movement and wrist tilt) · `all`.
- **Features:** time only (mean, spread, min, max, IQR, jerk, per axis and for the magnitude, plus axis correlations), or time + frequency (energy in 0–1, 1–3, 3–8, 8–25 Hz, dominant frequency, spectral entropy).
- **Models:** Naive Bayes, logistic regression, k-nearest neighbours, SVM, random forest, gradient boosting, a small neural net.

## How the comparison is set up

- **Whole sessions are held out, never random windows.** Neighbouring windows overlap and look alike; a random split leaks them into both sides and inflates the score. Each combination is trained on all training sessions but one and scored on the one left out, rotating.
- **TEST is used once**, after the choice is made. The only other model scored on TEST is a fixed baseline (accelerometer only, simple features, Naive Bayes) that stands in for the reference paper's setting.
- The smoothing length for the live page is also chosen on training sessions only.
- With a single training session the script falls back to holding out blocks and says so at the top of the report; with no TEST session it sets the last training session aside (three or more needed) and says so.

## Tested (12 Nov, before any real labelled data existed)

On seven synthetic sessions written to a scratch folder: session discovery and roles, the quick and the full run (210 combinations, 3.5 min), the report and figures, and loading the saved model in a test receiver fed with the real pilot recording. Those numbers are meaningless and were not kept. Nothing here has been run on real labelled sessions yet.
