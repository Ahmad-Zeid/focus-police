# AGENTS.md

Context for teammates and for AI coding agents working in this repo.

## What this is

Focus Police, the final project for AUB CMPS 297 (Wearable Sensing). A wrist-worn
accelerometer and gyroscope classify desk activity (writing, typing, phone, idle, plus
drinking and calculator look-alikes). After two minutes out of focus the wearer gets a nudge.
A phone dashboard shows the current activity and a focus timeline.

Team: Ahmad Zeid, Ali Sleiman, Mohamad Alame.

**Final submission: November 17, 2026, 23:55.** Demo and presentation in the last week of the course.

## Hard constraints

1. **Follow the original proposal.** The professor penalises projects that drift from it.
   Before cutting or changing anything, check it against the proposal. Only things the
   proposal lists as "optional extension" may be dropped (behaviour-change testing, EMG,
   standing/lying detection). Scope changes need the instructor's agreement.
2. **Our own data.** Course rule: new training and evaluation use student-collected data.
   Public datasets and pretrained models are allowed only as credited baselines.
3. **Synthetic data is for testing code only.** `data/synth`, `models/synth` and
   `results/synth` are git-ignored. No number from them goes in the report or slides.
4. **Honest evaluation.** Hold out whole sessions and whole people, report errors and
   limitations. There is no accuracy threshold to hit; 25% of the grade is explaining results.

## What the proposal commits us to

- Sensor: wrist accelerometer + gyroscope ("Apple Watch or similar"). Webcam for reference labels.
- Record: handwriting, typing, phone use (typing, scrolling, calling), drinking, idle, and
  look-alikes (bottle, cup, calculator), with the phone on the table and in hand.
- Different people and desks. Complete sessions reserved for evaluation.
- Compare a simple wrist-movement threshold with a trained classifier.
- Report accuracy per activity, false nudges per hour and nudge delay.
- Output: live current activity and focus timeline on the phone; watch vibration after two
  minutes out of focus.

## Status (November 12)

Done and tested on synthetic data:

- Loading, windowing, features, baseline, classifier, nudge counter.
- Evaluation (leave-one-session-out, leave-one-person-out) with a generated report.
- Live server, phone dashboard with a tone on nudge, labelled-recording page, replay tool.

Not done or not verified:

- **No real data yet.** The Apple Watch could not export data; the plan is a Galaxy Watch.
- The Sensor Logger input (`/data`) and the offline importer follow the documented formats
  but have **not been run against a real watch**. Test with a 5-minute recording first.
- Watch vibration: the dashboard plays a tone. `FP_NUDGE_WEBHOOK` can call a notification
  service so the watch buzzes; untested.
- Live streaming rate and lag from the Galaxy Watch are unknown.
- Report and slides.

## Next steps, in order

1. 5-minute test recording from the watch, run through the importer or the live server.
2. Full recording following `docs/PROTOCOL.md`.
3. `evaluate`, then `train`, on `data/raw`.
4. Live demo rehearsal with the real watch.
5. Report (PDF), slides, shared folder, submission email.

## Commands

```bash
uv sync
uv run pytest
uv run python -m focuspolice.evaluate --data data/raw --out results
uv run python -m focuspolice.train --data data/raw --out models/focuspolice.joblib
uv run uvicorn focuspolice.server:app --host 0.0.0.0 --port 8765
```

See `README.md` for the full workflow and the no-watch test path.

## Code map

| File | Role |
|---|---|
| `focuspolice/config.py` | Labels, classes, focus states, window and nudge settings. Proposal decisions live here. |
| `focuspolice/dataio.py` | Session folder format, loading, resampling to 50 Hz. |
| `focuspolice/features.py` | Windows and features. |
| `focuspolice/models.py` | Movement-threshold baseline, random forest, smoothing. |
| `focuspolice/nudge.py` | Off-task counter and nudge scoring. |
| `focuspolice/evaluate.py` | Held-out evaluation and `results/REPORT.md`. |
| `focuspolice/train.py` | Trains the model the live system loads. |
| `focuspolice/live.py`, `server.py` | Live engine, HTTP and WebSocket server, recorder. |
| `focuspolice/static/` | Phone dashboard and recorder pages. |
| `focuspolice/importer.py` | Offline watch export to a session folder. |
| `focuspolice/replay.py`, `synth.py` | Testing tools. |

## Conventions

- Python 3.12, managed with uv. Run `uv run pytest` before committing.
- Port 8765 (8000 clashes with Docker on some of our machines).
- Keep changes simple. The course says appropriate simple methods can earn full marks.
- Credit any reused code, model or dataset in the README.
