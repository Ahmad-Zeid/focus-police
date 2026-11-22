"""Which sensors does detection need?  (22 Nov 2026, added after the demo question "are you using all three sensors?")

    python sensors.py            # from Model/lab, about 2 minutes

Same evaluation as train.py version 2: every session predicted as a continuous stream every 0.5 s, one whole
session held out at a time, extra trees, 4 s windows, calibrated. Only the sensor set changes. Holding out
P1's session is the cross-person test (the model has only seen the other participant).

Sensor sets (what the watch records, see FocusWatch SensorService.kt):
  accelerometer        raw accelerometer only (gravity and movement mixed): the reference paper's sensor
  gyroscope            rotation rate only
  accel + gyro         the two physical sensors, raw
  accel split          Android's split of the accelerometer into movement (linear acceleration) and wrist
                       angle (gravity). These are fused virtual sensors, so they also lean on the gyroscope
  all (shipped base)   movement + gravity + gyroscope: what train.py's candidates use
  all + angle/context  the shipped model's full feature set (wrist angles, last 10/30 s)
The magnetometer is not recorded on its own; the watch's rotation vector (roll/pitch/yaw) uses it, and no
model uses that.

Writes sensors_results.csv next to this file.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import train as T  # noqa: E402

SETS = {
    "accelerometer": {"sensors": "acc", "freq": True, "calibrated": True},
    "gyroscope": {"sensors": "gyro", "freq": True, "calibrated": True},
    "accel + gyro (raw)": {"sensors": "acc+gyro", "freq": True, "calibrated": True},
    "accel split: movement + gravity": {"sensors": "motion+angle", "freq": True, "calibrated": True},
    "all: movement + gravity + gyro": {"sensors": "all", "freq": True, "calibrated": True},
    "all + wrist angle + context (shipped)": T.FULL,
}


def main():
    sessions = [s for s in T.load_sessions(T.SESSIONS) if s["role"] == "train"]
    print("held-out sessions:", [f"{s['name']}-{s['participant']}" for s in sessions])
    rows = []
    for name, cfg in SETS.items():
        cfg = {"window_s": 4.0, **cfg}
        tables = [T.table(s, cfg) for s in sessions]
        r = T.summarize(T.cross_validate("Extra trees", tables))
        row = {"sensors": name, "balanced": round(r["balanced"], 3), "phone_caught": round(r["caught"], 3),
               "false_alarm": round(r["false_alarm"], 3), "false_nudges": r["false_nudges"],
               "activity_macro_f1": round(r["macro_f1"], 3)}
        for s, b in zip(sessions, r["per_held_out"]):
            row[f"held out {s['name']}-{s['participant']}"] = b
        rows.append(row)
        print(f"{row['balanced']:.3f}  false alarms {row['false_alarm']:.3f}  per held-out {r['per_held_out']}  {name}",
              flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "sensors_results.csv"), index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
