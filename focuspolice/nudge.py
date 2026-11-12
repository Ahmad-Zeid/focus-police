"""Off-task counter and nudge scoring.

The counter rises while the wearer is off-task, drains while on-task and holds
still while neutral (drinking), so one sip does not reset a phone break. It
fires a nudge when it reaches NUDGE_AFTER_S and then starts again from zero.
"""

from __future__ import annotations

from .config import DRAIN_RATE, NUDGE_AFTER_S, NUDGE_MATCH_TOL_S


class NudgeCounter:
    def __init__(self, nudge_after_s: float = NUDGE_AFTER_S, drain_rate: float = DRAIN_RATE):
        self.nudge_after_s = nudge_after_s
        self.drain_rate = drain_rate
        self.value = 0.0

    def update(self, focus: str, dt: float) -> bool:
        """Advance by dt seconds in the given focus state. True if a nudge fires."""
        if focus == "off":
            self.value += dt
        elif focus == "on":
            self.value = max(0.0, self.value - self.drain_rate * dt)
        if self.value >= self.nudge_after_s:
            self.value = 0.0
            return True
        return False


def nudge_times(times, focus_states, dt: float, **kwargs) -> list[float]:
    """Times at which nudges fire for a sequence of focus states spaced dt apart."""
    counter = NudgeCounter(**kwargs)
    return [float(t) for t, f in zip(times, focus_states) if counter.update(f, dt)]


def score_nudges(predicted, reference, duration_s: float, tol_s: float = NUDGE_MATCH_TOL_S) -> dict:
    """Match predicted nudges to reference nudges (those the true labels produce).

    false nudge = a predicted nudge with no reference nudge within tol_s
    missed      = a reference nudge with no predicted nudge within tol_s
    delay       = predicted time minus reference time, for matched pairs
    """
    unmatched = list(predicted)
    delays, missed = [], 0
    for ref in reference:
        near = [p for p in unmatched if abs(p - ref) <= tol_s]
        if not near:
            missed += 1
            continue
        best = min(near, key=lambda p: abs(p - ref))
        unmatched.remove(best)
        delays.append(best - ref)
    hours = duration_s / 3600.0
    return {
        "reference_nudges": len(reference),
        "predicted_nudges": len(predicted),
        "matched": len(delays),
        "missed": missed,
        "false_nudges": len(unmatched),
        "false_nudges_per_hour": len(unmatched) / hours if hours > 0 else 0.0,
        "mean_delay_s": sum(delays) / len(delays) if delays else None,
        "delays_s": delays,
        "hours": hours,
    }
