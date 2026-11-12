"""Replay a recorded session into a receiver as if the watch were sending it live.
For testing the live page and the model without the watch (e.g. rehearsing the demo).

    python3 replay.py "../Data Collection/sessions/LIVE_20261112-081101/stream.csv" [http://localhost:8000] [--from SECONDS]

Sends 1 s batches in real time, with timestamps moved to now. The receiver saves it as a new
session called REPLAY_<time>, so point it at a scratch folder if you don't want that in sessions/.
"""
import csv
import json
import sys
import time
import urllib.request

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = args[0]
url = (args[1] if len(args) > 1 else "http://localhost:8000").rstrip("/") + "/ingest"
start_at = float(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0.0

with open(path) as f:
    r = csv.reader(f)
    cols = next(r)
    rows = [[float(v) for v in row] for row in r]
t0 = rows[0][0] / 1e9
rows = [[x[0] / 1e9 - t0] + x[1:] for x in rows if x[0] / 1e9 - t0 >= start_at]
session = "REPLAY_" + time.strftime("%Y%m%d-%H%M%S")
shift = time.time() - rows[0][0]
print(f"replaying {len(rows)} rows ({rows[-1][0] - rows[0][0]:.0f} s) to {url} as {session}")
i = 0
while i < len(rows):
    j = i
    while j < len(rows) and rows[j][0] + shift <= time.time():
        j += 1
    if j > i:
        batch = [[x[0] + shift] + x[1:] for x in rows[i:j]]
        body = json.dumps({"session": session, "cols": cols, "rows": batch, "device": "replay", "hz": 50}).encode()
        try:
            urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json"}), timeout=5)
            i = j
        except Exception as e:
            print("send failed:", e)
    time.sleep(1.0)
print("done")
