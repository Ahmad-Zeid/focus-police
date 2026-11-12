# For AI assistants working in this repository

`HANDOFF.md` describes what exists, how each part was tested, what was tried and dropped, and what is still undecided. `README.md` is the short version.

Context that is easy to miss:

- There is one labelled session and no test session yet. The scores in `Model/REPORT.md` come from holding out parts of that same session, so they are optimistic.
- Evaluation is done by whole sessions, and the TEST session is not used to choose anything (`HANDOFF.md`, section 9).
- `Model/features.py` is shared by training and the live page; `WristStream/model.pkl` has to be retrained after it changes.
- `stream.csv` files are the raw recordings. `labelled.csv`, `segments.csv` and `check.png` are derived and can be regenerated with `segment.py`. Hand corrections to labels are noted in the session's `NOTES.md`.
- The four top-level folders reference each other by relative path.
- Without a watch, `WristStream/replay.py` feeds a recording into a receiver as if it were live. A second port and a scratch output folder keep replays out of `sessions/`.
- The Apple Watch app was dropped (`HANDOFF.md`, section 11).
- Two design questions are open for the team: which hand writes, and whether idle counts as on task (`HANDOFF.md`, section 12).
