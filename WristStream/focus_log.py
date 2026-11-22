"""The focus timeline of one recording: what the proposal calls "a focus timeline for the session".

receiver.py feeds it every live prediction; it keeps one point per second for the whole recording, notes when
the laptop nudged and when the watch buzzed, and sums the session up. /focus-data serves it to the focus page,
and the receiver saves the summary next to the recording as focus_summary.json.

Definitions (the same as the nudge simulation in Model/train.py):
  focused        a prediction of an on-task class (desk work, drinking, holding a cup)
  on the phone   the phone classes together at 0.5 or more
  distraction    phone for NUDGE_S seconds in a row, i.e. a laptop nudge; shorter glances are not counted
"""


class FocusLog:
    def __init__(self, session, nudge_s, watch_s):
        self.session, self.nudge_s, self.watch_s = session, nudge_s, watch_s
        self.points = []          # (t, on): on is True, False or None (no prediction), one per second
        self.nudges, self.buzzes = [], []
        self._off_since, self._nudged, self._buzzed = None, False, False

    def add(self, t, on):
        """One live prediction at watch time t (epoch seconds). Called every 0.5 s; keeps one per second."""
        if on is False:
            if self._off_since is None:
                self._off_since, self._nudged, self._buzzed = t, False, False
            if not self._nudged and t - self._off_since >= self.nudge_s:
                self.nudges.append(t); self._nudged = True
            if not self._buzzed and t - self._off_since >= self.watch_s:
                self.buzzes.append(t); self._buzzed = True
        else:
            self._off_since = None
        if not self.points or t - self.points[-1][0] >= 1.0:
            self.points.append((t, on))

    def gap(self, t):
        """The watch went quiet: no prediction from here (shown grey, ends any phone stretch)."""
        self._off_since = None
        if self.points and self.points[-1][1] is not None:
            self.points.append((t, None))

    def runs(self):
        """[start_s, end_s, state] from the start of the recording; state 1 focused, 0 phone, -1 no data."""
        if not self.points:
            return []
        t0, out = self.points[0][0], []
        for (t, on), nxt in zip(self.points, self.points[1:] + [None]):
            end = nxt[0] if nxt else t + 1.0
            if end - t > 3.0:                      # a hole in the stream
                end = t + 1.0
            s = -1 if on is None else int(on)
            if out and out[-1][2] == s and abs(out[-1][1] - (t - t0)) < 0.01:
                out[-1][1] = end - t0              # continues the previous stretch
            else:
                out.append([t - t0, end - t0, s])
        return [[round(a, 1), round(b, 1), s] for a, b, s in out]

    def summary(self):
        runs = self.runs()
        focus = sum(e - s for s, e, k in runs if k == 1)
        phone = sum(e - s for s, e, k in runs if k == 0)
        longest = max([e - s for s, e, k in runs if k == 1] or [0])
        t0 = self.points[0][0] if self.points else 0
        back = []                                   # after each nudge: how long until focused again
        for n in self.nudges:
            later = [s for s, e, k in runs if k == 1 and s >= n - t0]
            back.append(round(later[0] - (n - t0), 1) if later else None)
        return {"session": self.session, "length_s": round(runs[-1][1], 1) if runs else 0,
                "focus_s": round(focus, 1), "phone_s": round(phone, 1),
                "focus_share": round(focus / (focus + phone), 3) if focus + phone else None,
                "distractions": len(self.nudges), "watch_buzzes": len(self.buzzes),
                "longest_focus_s": round(longest, 1),
                "nudges_at_s": [round(n - t0, 1) for n in self.nudges],
                "buzzes_at_s": [round(b - t0, 1) for b in self.buzzes],
                "back_to_work_s": back,
                "nudge_after_s": self.nudge_s, "watch_buzz_after_s": self.watch_s}
