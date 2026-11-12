"""Live engine: buffers incoming samples, classifies every second, runs the nudge counter.

All timing follows the sensor's own timestamps, so a recording replayed faster
than real time behaves exactly like the live stream.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from .config import FS, LIVE_HOP_S, NUDGE_AFTER_S, SMOOTH_WINDOWS, WINDOW_S, focus_of
from .dataio import to_ms2, write_session
from .features import compute_features
from .models import load_bundle, smooth_labels
from .nudge import NudgeCounter

BUFFER_S = 30.0
MIN_RATE_HZ = 10.0


class Stream:
    """Time-stamped xyz samples from one sensor."""

    def __init__(self):
        self.t: deque = deque()
        self.v: deque = deque()

    def add(self, t, xyz):
        self.t.append(float(t))
        self.v.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))
        while self.t and self.t[0] < t - BUFFER_S:
            self.t.popleft()
            self.v.popleft()

    def clear(self):
        self.t.clear()
        self.v.clear()

    @property
    def latest(self):
        return self.t[-1] if self.t else None

    def on_grid(self, grid):
        """Interpolate onto grid; None if the stream does not cover it well enough."""
        if len(self.t) < 2:
            return None
        t, v = np.array(self.t), np.array(self.v)
        inside = (t >= grid[0] - 0.5) & (t <= grid[-1] + 0.5)
        if inside.sum() < MIN_RATE_HZ * (grid[-1] - grid[0]) or t[0] > grid[0] + 0.5:
            return None
        return np.column_stack([np.interp(grid, t, v[:, i]) for i in range(3)])


class Recorder:
    """Saves the incoming stream and the labels pressed on the /record page."""

    def __init__(self):
        self.active = False
        self.meta: dict = {}
        self.raw = {"acc": [], "gyr": [], "grav": []}
        self.events: list[tuple[float, str]] = []
        self.out_dir = None

    def start(self, out_dir, meta):
        self.active, self.meta, self.out_dir = True, meta, Path(out_dir)
        self.raw = {"acc": [], "gyr": [], "grav": []}
        self.events = []

    def add(self, stream, t, xyz):
        if self.active:
            self.raw[stream].append((t, *xyz))

    def label(self, t, label):
        if self.active and t is not None:
            self.events.append((float(t), label))

    def stop(self, acc_has_gravity: bool):
        self.active = False
        acc = np.array(self.raw["acc"], float)
        gyr = np.array(self.raw["gyr"], float)
        if len(acc) < 2 or len(gyr) < 2:
            raise ValueError("nothing recorded: no accelerometer or gyroscope samples arrived")
        acc, gyr = acc[np.argsort(acc[:, 0])], gyr[np.argsort(gyr[:, 0])]
        t = acc[:, 0]
        a = acc[:, 1:4].copy()
        if not acc_has_gravity:
            grav = np.array(self.raw["grav"], float)
            if len(grav) < 2:
                raise ValueError("acceleration has no gravity and no gravity stream was received")
            grav = grav[np.argsort(grav[:, 0])]
            a += np.column_stack([np.interp(t, grav[:, 0], grav[:, i]) for i in (1, 2, 3)])
        g = np.column_stack([np.interp(t, gyr[:, 0], gyr[:, i]) for i in (1, 2, 3)])
        t0 = t[0]
        intervals = []
        for (start, label), nxt in zip(self.events, self.events[1:] + [(t[-1], "none")]):
            if label != "none" and nxt[0] > start:
                intervals.append((start - t0, nxt[0] - t0, label))
        write_session(self.out_dir, t - t0, np.column_stack([to_ms2(a), g]), intervals, self.meta)
        seconds = float(t[-1] - t0)
        return {"path": str(self.out_dir), "seconds": seconds, "samples": int(len(t)),
                "rate_hz": len(t) / seconds if seconds else 0.0, "labels": len(intervals)}


class LiveEngine:
    def __init__(self, model_path=None):
        self.acc, self.gyr, self.grav = Stream(), Stream(), Stream()
        self.acc_has_gravity = True
        self.total_seen = False     # a "total acceleration" stream has arrived
        self.user_acc_seen = False  # a plain "accelerometer" stream has arrived
        self.bundle = None
        self.model_info = None
        if model_path and Path(model_path).exists():
            self.bundle = load_bundle(model_path)
            self.bundle["classifier"].model.n_jobs = 1  # one window at a time: threads only add overhead
            self.model_info = self.bundle["info"]
        self.recorder = Recorder()
        self.reset()

    def reset(self):
        self.counter = NudgeCounter()
        self.timeline: list[dict] = []
        self.recent: deque = deque(maxlen=SMOOTH_WINDOWS)
        self.next_t = None
        self.nudges = 0
        self.pending_nudge = False
        self.current = {"activity": None, "raw_activity": None, "focus": "neutral", "baseline_focus": None}
        self._arrivals: deque = deque(maxlen=400)

    # ---- input -------------------------------------------------------------

    def add(self, stream: str, t: float, xyz, wall: float):
        latest = self.acc.latest
        if stream == "acc" and latest is not None and (t < latest - 5 or t > latest + 60):
            for s in (self.acc, self.gyr, self.grav):  # clock jumped: new stream
                s.clear()
            self.next_t = None
        {"acc": self.acc, "gyr": self.gyr, "grav": self.grav}[stream].add(t, xyz)
        self.recorder.add(stream, t, xyz)
        if stream == "acc":
            self._arrivals.append((wall, t))

    @property
    def sensor_now(self):
        if self.acc.latest is None or self.gyr.latest is None:
            return self.acc.latest
        return min(self.acc.latest, self.gyr.latest)

    def rate_hz(self):
        if len(self._arrivals) < 2:
            return 0.0
        span = self._arrivals[-1][1] - self._arrivals[0][1]
        return (len(self._arrivals) - 1) / span if span > 0 else 0.0

    def last_arrival(self):
        return self._arrivals[-1][0] if self._arrivals else None

    # ---- decisions ---------------------------------------------------------

    def _window(self, end_t):
        grid = end_t - WINDOW_S + np.arange(int(WINDOW_S * FS)) / FS
        acc, gyr = self.acc.on_grid(grid), self.gyr.on_grid(grid)
        if acc is None or gyr is None:
            return None
        if not self.acc_has_gravity:
            grav = self.grav.on_grid(grid)
            if grav is None:
                return None
            acc = acc + grav
        return np.column_stack([to_ms2(acc), gyr])

    def process(self):
        """Make every decision that is due. Returns the new timeline entries."""
        now = self.sensor_now
        if now is None or self.gyr.latest is None:
            return []
        if self.next_t is None:
            self.next_t = now
        if now - self.next_t > BUFFER_S - WINDOW_S:  # fell too far behind
            self.next_t = now
        new = []
        while self.next_t <= now:
            end_t = self.next_t
            self.next_t += LIVE_HOP_S
            w = self._window(end_t)
            if w is None or self.bundle is None:
                continue
            X, _ = compute_features(w[None])
            raw = str(self.bundle["classifier"].predict(X)[0])
            self.recent.append(raw)
            activity = str(smooth_labels(list(self.recent))[-1])
            focus = focus_of(activity)
            if self.counter.update(focus, LIVE_HOP_S):
                self.nudges += 1
                self.pending_nudge = True
            self.current = {
                "activity": activity, "raw_activity": raw, "focus": focus,
                "baseline_focus": str(self.bundle["baseline"].predict_focus(X)[0]),
            }
            entry = {"t": round(end_t, 2), "activity": activity, "focus": focus}
            self.timeline.append(entry)
            new.append(entry)
        if len(self.timeline) > 7200:
            self.timeline = self.timeline[-7200:]
        return new

    def state(self, wall: float):
        last = self.last_arrival()
        age = None if last is None else wall - last
        nudge, self.pending_nudge = self.pending_nudge, False
        return {
            "type": "state",
            "connected": age is not None and age < 3.0,
            "sample_age_s": age,
            "rate_hz": round(self.rate_hz(), 1),
            "has_gyro": self.gyr.latest is not None,
            "has_model": self.bundle is not None,
            "sensor_t": self.sensor_now,
            **self.current,
            "counter_s": round(self.counter.value, 1),
            "nudge_after_s": NUDGE_AFTER_S,
            "nudges": self.nudges,
            "nudge": nudge,
            "recording": self.recorder.active,
            "record_label": self.recorder.events[-1][1] if self.recorder.events else None,
        }
