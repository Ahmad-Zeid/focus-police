"""Replay a recorded session into the live server, as if the sensor were streaming.

    python -m focuspolice.replay data/raw/ahmad_unscripted --speed 10

Useful for testing the dashboard without the watch and as a demo fallback.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

import pandas as pd

from .dataio import IMU_COLS


def replay(session_dir: str, url: str, speed: float, batch_s: float = 1.0):
    df = pd.read_csv(f"{session_dir}/imu.csv")
    base = time.time()
    t = df["t"].to_numpy() - df["t"].iloc[0]
    rows = df[IMU_COLS].to_numpy()
    start, wall0 = 0, time.time()
    while start < len(t):
        end = start
        while end < len(t) and t[end] < t[start] + batch_s:
            end += 1
        samples = [[base + float(t[i]), *map(float, rows[i])] for i in range(start, end)]
        req = urllib.request.Request(url, data=json.dumps({"samples": samples}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        start = end
        if start < len(t):
            wait = t[start] / speed - (time.time() - wall0)
            if wait > 0:
                time.sleep(wait)
    print(f"replayed {t[-1]:.0f} s of data at {speed:g}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session")
    ap.add_argument("--url", default="http://localhost:8765/ingest")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    replay(args.session, args.url, args.speed)


if __name__ == "__main__":
    main()
