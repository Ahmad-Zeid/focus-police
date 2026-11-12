# Focus Police

A watch on the wrist streams motion to a laptop, which works out what you are doing at your desk and nudges you after 10 seconds on your phone. CMPS 297AH Wearable Sensing final project, due Tue 17 Nov 2026, 23:55.

[`HANDOFF.md`](HANDOFF.md) has the full account: what was built, how it was tested, and why each decision was made.

## Where things stand (13 Nov)

- The watch app, the laptop receiver, the live page, the session runner, automatic labelling and the training pipeline all work.
- There is only **one labelled recording** (8 activities, about a minute each, one person). The model trained on it is a first try. Tried live, it mixed up holding the phone with drinking and typing with writing, so more data looks necessary.
- Recording so far used one borrowed Galaxy Watch5 Pro. The app installs on any Wear OS 3+ watch with `FocusWatch/setup-watch.sh`. Training, analysis and replay need no watch.

## The five commands that matter

Set up Python once (3.10 or newer):

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Train on all labelled sessions and write `Model/REPORT.md`:

```bash
cd Model && python3 train.py
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
