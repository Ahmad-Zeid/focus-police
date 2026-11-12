# Recordings

One folder per watch recording, named `LIVE_<watch start time>`. All: Galaxy Watch5 Pro, left wrist, right-handed person, sitting, 12 Nov 2026, about 50 Hz, no gaps.

| Folder | Length | Labelled | What it is |
|---|---|---|---|
| `LIVE_20261112-080550` | 13 s | no | setup connection test |
| `LIVE_20261112-081101` | 5.5 min | no | first real recording, free mix of activities; `pilot-unlabelled-check.png` shows four clusters found without labels |
| `LIVE_20261112-082701` | 10 s | no | short test |
| `LIVE_20261112-083255` | 2.9 min | no | recorded without the runner; contents unknown |
| `LIVE_20261112-130029` | 2.6 min | no | recorded without the runner; contents unknown |
| `LIVE_20261112-233008` | 13 s | no | connection test |
| `LIVE_20261112-233400` | 12.1 min | **yes (S1)** | the only labelled session: 8 blocks. See its `NOTES.md` |
| `LIVE_20261112-235320` | 4.2 min | no | trying the first model live |
| `LIVE_20261112-235824` | 1.3 min | no | same |

Files in a session folder: `stream.csv` (the raw recording, never modified), `meta.json` (device, OS, columns), and for labelled sessions the runner's timestamp file, `labelled.csv`, `segments.csv`, `check.png`. Column meanings and the labelling method are in `../../HANDOFF.md`, sections 5 and 7.

Add new recordings by recording with the runner at http://localhost:8000/runner; update this table when you do.
