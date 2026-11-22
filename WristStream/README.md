# WristStream: the laptop side

Receives the watch's motion stream, saves it, runs the models and serves the pages.

```
Galaxy Watch (FocusWatch app) --HTTP POST every 1 s--> receiver.py --> Data Collection/sessions/LIVE_<time>/stream.csv
                                                          |--> main model + instant model, every 0.5 s
                                                          |--> pages: /focus (phone), / (live), /runner
                                                          \--> reply to the watch: vibrate after 2 min off task
```

## Run

```bash
python3 receiver.py              # port 8000, saves to ../Data Collection/sessions
python3 receiver.py 8001         # another port (type <laptop address>:8001 on the watch)
python3 receiver.py 8010 /tmp/x  # another port and a scratch folder, e.g. for replays
```

It prints the laptop's address for the watch app and the phone. If macOS asks whether Python may accept incoming connections, click **Allow**. The watch and the laptop must be on the same Wi-Fi, and that network must let devices reach each other (campus Wi-Fi often does not; a phone hotspot works).

## Files

| File | What it does |
|---|---|
| `receiver.py` | HTTP server: `POST /ingest` from the watch, saves every sample, runs both models every 0.5 s, the 10 s nudge and the 2 min watch vibration, calibration (`POST /calibrate`), the session runner's log (`POST /runner-log`), the pages |
| `live_model.py` | loads `model.pkl` (main) and `model_instant.pkl` (instant); predicts and reports the class (desk work, drinking, holding a cup, phone) and the most likely activity inside it |
| `model.pkl`, `model_instant.pkl` | the trained models, written by `../Model/train.py` |
| `focus_log.py` | the session's focus timeline: one point per second, nudges, watch vibrations and the summary, saved as `focus_summary.json` with the recording |
| `focus.html` | the focus page at `/focus`, made for a phone |
| `live.html` | the live page at `/`: signals, probabilities, nudge banner |
| `replay.py` | plays a recording into a receiver as if the watch were sending it |

## The pages

- **Focus page** (`/focus`): FOCUSED or ON YOUR PHONE from the main model, with the activity under it (e.g. *Desk work: typing*); under that, **right now** from the instant model; while on the phone, countdowns to the nudge and to the watch vibration; the session's focus share, focus and phone time, distractions (10 s or more on the phone), longest focus stretch, a timeline of the whole session with the nudges marked, and how long each distraction lasted. It has a Calibrate button.
- **Live page** (`/`): three scrolling graphs (movement, rotation, wrist angle), the prediction and every class's probability, a 2-minute on/off strip, the nudge banner, the model's name and its test accuracy.
- **Session runner** (`/runner`): guides a labelled recording; see `../Data Collection/README.md`.

**Off task** means the phone classes together reach 0.5, the same rule `train.py` scores. The nudge fires after `NUDGE_S` = 10 s off task in a row; the watch vibrates after `WATCH_NUDGE_S` = 120 s (set `FOCUS_WATCH_NUDGE_S=20` to shorten it for a demo).

**Models:** the receiver reloads them within a second when `train.py` writes new ones. Without `model.pkl`, a hand-set placeholder runs and the page labels it in orange as not a result. A model that fails shows "model error"; recording continues.

## Testing without the watch

```bash
python3 receiver.py 8010 /tmp/focus-test
```

```bash
python3 replay.py "../Data Collection/sessions/LIVE_20261117-120625/stream.csv" http://localhost:8010 --from 500
```

Then open http://localhost:8010/focus and press Calibrate during a stretch of typing.
