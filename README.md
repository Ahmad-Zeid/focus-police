# Focus Police

A wrist sensor tells working from drifting: writing, typing, idle, or on your phone.
AUB CMPS 297 (Wearable Sensing) final project by Ahmad Zeid, Ali Sleiman and Mohamad Alame.

```
SENSE                    PROCESS ON COMPUTER                 SHOW ON PHONE
watch accel + gyro  -->  5 s windows -> classifier ->   -->  activity, focus timeline,
                         off-task counter                    nudge after 2 min out of focus
```

## Required hardware

- A wrist-worn accelerometer and gyroscope (smartwatch) that can stream or export data.
- A laptop on the same Wi-Fi network as the watch's phone.
- A phone with a browser for the dashboard.

## Setup

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
uv run pytest
```

## Workflow

1. **Record** labelled sessions into `data/raw/` following [docs/PROTOCOL.md](docs/PROTOCOL.md).
2. **Evaluate** with whole sessions and whole people held out:

   ```bash
   uv run python -m focuspolice.evaluate --data data/raw --out results
   ```

   This writes `results/REPORT.md`, `results/metrics.json` and confusion matrices.
3. **Train** the model the live system uses:

   ```bash
   uv run python -m focuspolice.train --data data/raw --out models/focuspolice.joblib
   ```

4. **Run live**:

   ```bash
   uv run uvicorn focuspolice.server:app --host 0.0.0.0 --port 8765
   ```

   - Dashboard: `http://<laptop-ip>:8765` on the phone. Tap "Enable sound" once.
   - Recorder: `http://<laptop-ip>:8765/record`.
   - Sensor Logger: HTTP Push to `http://<laptop-ip>:8765/data`.
   - Anything else: POST `{"samples": [[t, ax, ay, az, gx, gy, gz], ...]}` to `/ingest`.

## Trying it without a watch

Synthetic data exercises the code only. Its numbers mean nothing and never go in the report.

```bash
uv run python -m focuspolice.synth --out data/synth
```

```bash
uv run python -m focuspolice.train --data data/synth --out models/synth/focuspolice.joblib
```

```bash
FP_MODEL=models/synth/focuspolice.joblib uv run uvicorn focuspolice.server:app --port 8765
```

```bash
uv run python -m focuspolice.replay data/synth/p1_unscripted --speed 10
```

## Method

- **Windows:** 5 s at 50 Hz, 2.5 s hop offline, one decision per second live.
- **Features:** per-axis statistics, gravity direction (wrist pose), movement energy,
  jerk, spectral band energies, axis correlations. See `focuspolice/features.py`.
- **Baseline:** a single wrist-movement threshold. Little movement means out of focus.
- **Classifier:** random forest over the features, predicting the desk activity.
- **Nudge:** a counter rises while off-task (phone, idle), drains while on-task
  (writing, typing, calculator) and holds still while drinking. A nudge fires at 2 minutes.
- **Evaluation:** leave-one-session-out and leave-one-person-out. Accuracy per activity,
  confusion matrix, off-task detection for baseline vs classifier, false nudges per hour
  and nudge delay.

Class and focus definitions are in `focuspolice/config.py`.

## Layout

```
focuspolice/   config, dataio, features, models, nudge, evaluate, train,
               live + server (live system), importer, replay, synth
focuspolice/static/   dashboard.html, record.html
docs/PROTOCOL.md      how to record
data/raw/             recorded sessions
results/              evaluation output
models/               trained model files
```

## Credits

scikit-learn (random forest), FastAPI and uvicorn (server), NumPy, pandas, SciPy,
matplotlib. Research starting point: Mosin, Min and Salamy, "Using a Wrist-Worn
Accelerometer to Detect Off-Task Activities While Studying or Working" (2025),
doi:10.1007/978-3-031-94953-1_12. Code written with assistance from Claude (Anthropic).
