"""Live server: receives sensor data, classifies, serves the phone dashboard.

    uv run uvicorn focuspolice.server:app --host 0.0.0.0 --port 8765

Open http://<laptop-ip>:8765 on the phone for the dashboard and
http://<laptop-ip>:8765/record for the labelled-recording page.

Environment variables:
    FP_MODEL          model file (default models/focuspolice.joblib)
    FP_DATA_DIR       where /record saves sessions (default data/raw)
    FP_NAME_FILTER    only accept Sensor Logger streams whose name contains this
                      (for example "watch" when the phone also sends its own sensors)
    FP_NUDGE_WEBHOOK  URL that receives a POST on every nudge (for example a
                      notification service that makes the watch vibrate)
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .config import CLASSES, FINE_TO_CLASS, FOCUS
from .live import LiveEngine

STATIC = Path(__file__).parent / "static"
MODEL_PATH = os.environ.get("FP_MODEL", "models/focuspolice.joblib")
DATA_DIR = os.environ.get("FP_DATA_DIR", "data/raw")
NAME_FILTER = os.environ.get("FP_NAME_FILTER", "").lower()
WEBHOOK = os.environ.get("FP_NUDGE_WEBHOOK", "")

engine = LiveEngine(MODEL_PATH)
clients: set[WebSocket] = set()


def send_webhook():
    def post():
        try:
            req = urllib.request.Request(WEBHOOK, data=b"Focus Police: back to work", method="POST")
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:  # the demo must not die because a notification failed
            print(f"nudge webhook failed: {exc}")
    threading.Thread(target=post, daemon=True).start()


async def broadcast(message: dict):
    for ws in list(clients):
        try:
            await ws.send_json(message)
        except Exception:
            clients.discard(ws)


async def ticker():
    while True:
        await asyncio.sleep(0.25)
        new = engine.process()
        state = engine.state(time.time())
        state["new"] = new
        if state["nudge"] and WEBHOOK:
            send_webhook()
        await broadcast(state)


@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(ticker())
    yield
    task.cancel()


app = FastAPI(title="Focus Police", lifespan=lifespan)


@app.get("/")
def dashboard():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/record")
def record_page():
    return FileResponse(STATIC / "record.html")


@app.get("/api/config")
def config():
    return {"classes": CLASSES, "focus": FOCUS, "fine_labels": list(FINE_TO_CLASS),
            "model": engine.model_info}


@app.get("/api/state")
def state():
    return engine.state(time.time())


@app.post("/api/reset")
def reset():
    engine.reset()
    return {"ok": True}


# ---- sensor input -----------------------------------------------------------

def _seconds(t):
    """Sensor Logger sends nanoseconds since epoch; accept ms and s too."""
    t = float(t)
    if t > 1e17:
        return t / 1e9
    if t > 1e14:
        return t / 1e6
    if t > 1e11:
        return t / 1e3
    return t


@app.post("/data")
async def sensor_logger(request: Request):
    """Sensor Logger "HTTP Push": {"payload": [{"name", "time", "values": {x,y,z}}]}."""
    body = await request.json()
    wall, n = time.time(), 0
    for item in body.get("payload", []):
        name = str(item.get("name", "")).lower()
        v = item.get("values") or {}
        if NAME_FILTER and NAME_FILTER not in name:
            continue
        t = _seconds(item.get("time", 0))
        if "rotationRateX" in v:  # Apple Watch "wrist motion" record, acceleration in g
            g = (v["gravityX"], v["gravityY"], v["gravityZ"])
            a = (v["accelerationX"] + g[0], v["accelerationY"] + g[1], v["accelerationZ"] + g[2])
            engine.acc_has_gravity = True
            engine.add("acc", t, a, wall)
            engine.add("gyr", t, (v["rotationRateX"], v["rotationRateY"], v["rotationRateZ"]), wall)
            n += 1
            continue
        if not all(k in v for k in "xyz"):
            continue
        xyz = (v["x"], v["y"], v["z"])
        if "gyro" in name:
            if "uncal" in name:
                continue
            engine.add("gyr", t, xyz, wall)
        elif "gravity" in name:
            engine.add("grav", t, xyz, wall)
        elif "total" in name:
            engine.total_seen = True
            engine.acc_has_gravity = True
            engine.add("acc", t, xyz, wall)
        elif "accel" in name and "uncal" not in name:
            if engine.total_seen:
                continue  # total acceleration is preferred when both arrive
            engine.add("acc", t, xyz, wall)
            engine.user_acc_seen = True
        else:
            continue
        n += 1
    # Sensor Logger's plain "accelerometer" may exclude gravity. At a desk the wrist
    # barely accelerates, so a median magnitude near zero means gravity was removed
    # (about 1 means g units with gravity, about 9.8 means m/s^2 with gravity).
    if engine.user_acc_seen and not engine.total_seen:
        vals = list(engine.acc.v)[-100:]
        if len(vals) >= 50:
            mags = sorted((x * x + y * y + z * z) ** 0.5 for x, y, z in vals)
            engine.acc_has_gravity = mags[len(mags) // 2] > 0.5
    return {"ok": True, "accepted": n}


@app.post("/ingest")
async def ingest(request: Request):
    """Generic input: {"samples": [[t, ax, ay, az, gx, gy, gz], ...]}, gravity included."""
    body = await request.json()
    wall = time.time()
    engine.acc_has_gravity = True
    for s in body.get("samples", []):
        engine.add("acc", s[0], s[1:4], wall)
        engine.add("gyr", s[0], s[4:7], wall)
    return {"ok": True, "accepted": len(body.get("samples", []))}


# ---- labelled recording -----------------------------------------------------

def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "x"


@app.post("/api/record/start")
async def record_start(request: Request):
    meta = await request.json()
    if engine.recorder.active:
        raise HTTPException(409, "a recording is already running")
    if not meta.get("person") or not meta.get("session"):
        raise HTTPException(400, "person and session are required")
    out = Path(DATA_DIR) / f"{_slug(meta['person'])}_{_slug(meta['session'])}"
    if out.exists():
        raise HTTPException(409, f"{out} already exists; pick another session name")
    meta["person"] = _slug(meta["person"])
    meta["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    engine.recorder.start(out, meta)
    return {"ok": True, "path": str(out)}


@app.post("/api/record/label")
async def record_label(request: Request):
    label = (await request.json()).get("label")
    if label != "none" and label not in FINE_TO_CLASS:
        raise HTTPException(400, f"unknown label {label}")
    if not engine.recorder.active:
        raise HTTPException(409, "not recording")
    if engine.acc.latest is None:
        raise HTTPException(409, "no sensor data is arriving yet")
    engine.recorder.label(engine.acc.latest, label)
    return {"ok": True}


@app.post("/api/record/stop")
def record_stop():
    if not engine.recorder.active:
        raise HTTPException(409, "not recording")
    try:
        return engine.recorder.stop(engine.acc_has_gravity)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({"type": "init", "timeline": engine.timeline,
                               "classes": CLASSES, "focus": FOCUS})
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)
