"""Laptop side of WristStream: receives the watch's motion batches over HTTP.

Run:
    python3 receiver.py            # listens on port 8000
Then open http://localhost:8000 for the live page: connection, live graphs, and the
current prediction from live_model.py (a labelled placeholder until a model is trained).

Every watch session is saved to ../Data Collection/sessions/LIVE_<session>/stream.csv
with column names segment.py already understands (time in ns, userAcceleration*, gravity*,
rotationRate* ...), so a live recording can be labelled exactly like any other session.
The classifier lives in live_model.py and is reloaded whenever it or model.pkl changes.
"""
import csv
import importlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

import live_model

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
HERE = os.path.dirname(os.path.abspath(__file__))
# 2nd argument: where to save sessions (e.g. on a server). Default: the project's sessions folder.
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "Data Collection", "sessions")
COLS = ["time", "userAccelerationX", "userAccelerationY", "userAccelerationZ",
        "gravityX", "gravityY", "gravityZ", "rotationRateX", "rotationRateY", "rotationRateZ",
        "roll", "pitch", "yaw"]          # Apple Watch (WristStream); the Galaxy app sends its own "cols"

state = {"last_seen": 0.0, "session": None, "device": None, "total": 0, "files": {}, "cols": COLS}
recent = deque(maxlen=50 * 15)          # last 15 s of rows: the live graphs and the classifier window
NUDGE_S = 10                            # off-task for this long in a row -> the page shows the nudge
pred = {"model": None, "test_accuracy": None, "label": None, "probs": {}, "on_task": None,
        "off_since": None, "error": None, "t": None}
history = deque(maxlen=2 * 120)         # (time, label, confidence, on_task), 2 per second for 2 min
PLOT = ["userAccelerationX", "userAccelerationY", "userAccelerationZ",
        "rotationRateX", "rotationRateY", "rotationRateZ", "gravityX", "gravityY", "gravityZ"]
arrivals = deque(maxlen=60)             # (wall time, n rows) per request, for the rate
lock = threading.Lock()


def classifier_loop():
    """Every 0.5 s: run live_model on the last WINDOW_S seconds. Reloads the model when its files change."""
    global live_model
    model, stamp = None, None
    while True:
        time.sleep(0.5)
        files = [live_model.__file__, live_model.MODEL_PATH]
        now_stamp = tuple(os.path.getmtime(f) if os.path.exists(f) else 0 for f in files)
        try:
            if now_stamp != stamp:
                live_model = importlib.reload(live_model)
                model, stamp = live_model.load(), now_stamp
                print(f"\n[model] {model.name}")
                pred.update(model=model.name, test_accuracy=model.test_accuracy,
                            accuracy_kind=getattr(model, "accuracy_kind", ""), error=None)
            with lock:
                fresh = state["last_seen"] and time.time() - state["last_seen"] < 3
                cols = state["cols"]
                n = int(model.window_s * 50)
                rows = list(recent)[-n:] if fresh and len(recent) >= n else None
            if rows is None:
                pred.update(label=None, probs={}, on_task=None, off_since=None)
                continue
            a = np.array(rows, dtype=float)
            win = {c: a[:, i] for i, c in enumerate(cols) if i < a.shape[1]}
            t = rows[-1][0]
            if t == pred["t"]:                      # no new samples since the last prediction
                continue
            label, probs = model.predict(win)
            on = live_model.is_on_task(label)
            off_since = pred["off_since"] if on is False else None
            if on is False and off_since is None:
                off_since = t
            pred.update(label=label, probs=probs, on_task=on, off_since=off_since, t=t, error=None)
            history.append((t, label, round(probs.get(label, 0), 3), on))
        except Exception as e:                      # a broken model must never stop the recording
            pred.update(error=f"{type(e).__name__}: {e}", label=None)
            stamp = now_stamp


def write_rows(session, rows, cols=None, meta=None):
    path = state["files"].get(session)
    if path is None:
        folder = os.path.join(OUT, f"LIVE_{session}")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "stream.csv")
        new = not os.path.exists(path)
        state["files"][session] = path
        if new:
            with open(path, "w") as f:
                f.write(",".join(cols or COLS) + "\n")
            with open(os.path.join(folder, "meta.json"), "w") as f:
                json.dump({"session": session, "cols": cols or COLS, **(meta or {}),
                           "first_seen": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        print(f"\n[new session] {session} -> {os.path.relpath(path, HERE)}")
    with open(path, "a") as f:
        for r in rows:
            f.write(f"{int(r[0] * 1e9)}," + ",".join(f"{v:.6f}" for v in r[1:]) + "\n")


def merge_runner(old_text, new_text):
    """Two runner files for ONE watch recording (e.g. a session ended early by mistake, then restarted
    without stopping the watch): keep the blocks of both. The newer file's block numbers are shifted
    past the older file's so they stay distinct."""
    old = list(csv.DictReader(io.StringIO(old_text)))
    new = list(csv.DictReader(io.StringIO(new_text)))
    if not old:
        return new_text
    shift = max([int(float(r["index"])) for r in old if r.get("index") not in (None, "")] + [-1]) + 1
    for r in new:
        if r.get("index") not in (None, "") and int(float(r["index"])) >= 0:
            r["index"] = str(int(float(r["index"])) + shift)
    fields = list(dict.fromkeys(list(old[0].keys()) + list(new[0].keys() if new else [])))
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fields, restval="")
    w.writeheader()
    w.writerows(old + new)
    return out.getvalue()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def runner_log(self, url):
        """The runner hands over its timestamp file at the end of a session: file it next to the
        watch recording it belongs to (the session streaming now, or the last one) and label it."""
        n = int(self.headers.get("Content-Length", 0))
        text = self.rfile.read(n).decode("utf-8", "replace")
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", parse_qs(url.query).get("name", ["runner.csv"])[0])
        if "runner" not in name:
            name = name.replace(".csv", "") + "_runner.csv"
        with lock:
            session = state["session"]
        if not session:
            self.send_json({"saved": False, "error": "no watch recording has arrived since the receiver started"}); return
        folder = os.path.join(OUT, f"LIVE_{session}")
        os.makedirs(folder, exist_ok=True)
        merged = False
        for old in sorted(os.listdir(folder)):              # a second runner file for the same recording:
            if old.endswith("_runner.csv"):                 # keep both sets of blocks, and the original file
                with open(os.path.join(folder, old)) as f:
                    text = merge_runner(f.read(), text)
                k = sum(x.startswith(old + ".part") for x in os.listdir(folder)) + 1
                os.replace(os.path.join(folder, old), os.path.join(folder, f"{old}.part{k}"))
                merged = True
        with open(os.path.join(folder, name), "w") as f:
            f.write(text)
        out = {"saved": True, "folder": f"LIVE_{session}", "ok": False, "output": "", "merged": merged}
        if os.path.exists(SEGMENT):
            time.sleep(3)                                   # let the watch deliver the final claps
            try:
                p = subprocess.run([sys.executable, SEGMENT, folder], capture_output=True, text=True, timeout=180)
                out["ok"] = p.returncode == 0
                out["output"] = (p.stdout + p.stderr).strip()[-4000:]
            except Exception as e:
                out["output"] = f"segment.py did not finish: {e}"
        else:
            out["output"] = "segment.py not found next to the receiver; label it later."
        print(f"\n[runner] {name} -> LIVE_{session}  labelled={'yes' if out['ok'] else 'NO'}")
        self.send_json(out)

    def do_POST(self):
        if urlparse(self.path).path == "/runner-log":
            self.runner_log(urlparse(self.path)); return
        if self.path != "/ingest":
            self.send_error(404); return
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n))
            rows = body["rows"]
            session = body.get("session", "unknown")
        except Exception as e:
            self.send_error(400, str(e)); return
        with lock:
            write_rows(session, rows, body.get("cols"),
                       {k: body.get(k) for k in ("device", "os", "hz") if k in body})
            if session != state["session"]:
                recent.clear()
            state.update(last_seen=time.time(), session=session, device=body.get("device"),
                         cols=body.get("cols") or COLS)
            state["total"] += len(rows)
            recent.extend(rows)
            arrivals.append((time.time(), len(rows)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def send_json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/recent":
            # rows newer than ?since= (epoch s), only the plotted columns, oldest first
            since = float(parse_qs(url.query).get("since", ["0"])[0])
            with lock:
                cols = state["cols"]
                idx = [cols.index(c) for c in PLOT if c in cols]
                out = [[r[0]] + [r[i] for i in idx] for r in recent if r[0] > since]
            self.send_json({"cols": ["time"] + [c for c in PLOT if c in cols], "rows": out}); return
        if url.path == "/predict":
            p = dict(pred)
            p["off_for"] = round(p["t"] - p["off_since"], 1) if p.get("off_since") and p.get("t") else 0
            p["nudge"] = p["off_for"] >= NUDGE_S
            p["nudge_after"] = NUDGE_S
            p["history"] = list(history)
            self.send_json(p); return
        if url.path == "/status":
            with lock:
                now = time.time()
                win = [n for t, n in arrivals if now - t <= 5]
                last = recent[-1] if recent else None
                mag = None
                if recent:
                    tail = list(recent)[-50:]
                    mag = sum((r[1] ** 2 + r[2] ** 2 + r[3] ** 2) ** 0.5 for r in tail) / len(tail)
                s = {"connected": now - state["last_seen"] < 3, "seconds_since": round(now - state["last_seen"], 1) if state["last_seen"] else None,
                     "session": state["session"], "device": state["device"], "total": state["total"],
                     "rate_hz": round(sum(win) / 5, 1), "motion_g": round(mag, 3) if mag is not None else None,
                     "delay_s": round(now - last[0], 2) if last else None}
            data = json.dumps(s).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
            self.wfile.write(data); return
        if url.path == "/runner":
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            with open(RUNNER_PATH, "rb") as f:
                self.wfile.write(f.read())
            return
        if url.path == "/check":
            folder = os.path.basename(parse_qs(url.query).get("folder", [""])[0])
            png = os.path.join(OUT, folder, "check.png")
            if not folder.startswith("LIVE_") or not os.path.exists(png):
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            with open(png, "rb") as f:
                self.wfile.write(f.read())
            return
        if url.path == "/":
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            with open(PAGE_PATH, "rb") as f:
                self.wfile.write(f.read())
            return
        self.send_error(404)


DATA_DIR = os.path.join(HERE, "..", "Data Collection")
RUNNER_PATH = os.path.join(DATA_DIR, "session-runner.html")     # served at /runner
SEGMENT = os.path.join(DATA_DIR, "segment.py")
PAGE_PATH = os.path.join(HERE, "live.html")   # read on every request, so page edits need no restart


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1)); return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def ticker():
    while True:
        time.sleep(1)
        with lock:
            now = time.time()
            win = sum(n for t, n in arrivals if now - t <= 5) / 5
            age = now - state["last_seen"] if state["last_seen"] else None
        msg = f"rate {win:5.1f} Hz | total {state['total']:7d} | " + ("CONNECTED" if age is not None and age < 3 else "waiting for watch")
        print("\r" + msg + " " * 10, end="", flush=True)


if __name__ == "__main__":
    ip = lan_ip()
    print(f"WristStream receiver on http://{ip}:{PORT}  (put this address in the watch app)")
    print(f"live page: http://localhost:{PORT}   session runner: http://localhost:{PORT}/runner")
    print(f"saving to {os.path.normpath(OUT)}")
    threading.Thread(target=ticker, daemon=True).start()
    threading.Thread(target=classifier_loop, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
