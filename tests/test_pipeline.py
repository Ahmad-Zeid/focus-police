import numpy as np
from fastapi.testclient import TestClient

from focuspolice.config import NUDGE_AFTER_S
from focuspolice.dataio import load_all, load_session
from focuspolice.evaluate import evaluate
from focuspolice.features import session_windows
from focuspolice.nudge import NudgeCounter, nudge_times, score_nudges
from focuspolice.synth import generate
from focuspolice.train import train


def test_counter_fires_after_two_minutes_off_task():
    c = NudgeCounter()
    fired = [c.update("off", 1.0) for _ in range(int(NUDGE_AFTER_S))]
    assert fired[-1] and not any(fired[:-1])
    assert c.value == 0.0


def test_sip_does_not_reset_and_work_drains():
    c = NudgeCounter()
    for _ in range(100):
        c.update("off", 1.0)
    for _ in range(30):
        c.update("neutral", 1.0)
    assert c.value == 100
    for _ in range(40):
        c.update("on", 1.0)
    assert c.value == 60


def test_score_nudges_counts_false_and_missed():
    s = score_nudges(predicted=[130, 900], reference=[120, 500], duration_s=3600)
    assert s["matched"] == 1 and s["missed"] == 1 and s["false_nudges"] == 1
    assert s["false_nudges_per_hour"] == 1.0 and s["mean_delay_s"] == 10
    assert nudge_times([1, 2, 3], ["off"] * 3, 1.0, nudge_after_s=2) == [2.0]


def test_end_to_end_on_synthetic_data(tmp_path, monkeypatch):
    data = tmp_path / "data"
    generate(str(data), people=("a", "b"), seed=1)
    sessions = load_all(data)
    assert len(sessions) == 6
    w = session_windows(sessions[0])
    assert w["X"].shape[0] == len(w["y"]) and np.isfinite(w["X"]).all()

    res = evaluate(str(data), str(tmp_path / "results"))
    assert res["leave-one-person-out"]["activity"]["classifier"]["accuracy"] > 0.8
    assert (tmp_path / "results" / "REPORT.md").exists()

    model = tmp_path / "model.joblib"
    train(str(data), str(model))

    # Live path: stream one session into the server and record it back out.
    monkeypatch.setenv("FP_MODEL", str(model))
    monkeypatch.setenv("FP_DATA_DIR", str(tmp_path / "rec"))
    import importlib

    import focuspolice.server as server
    server = importlib.reload(server)
    client = TestClient(server.app)
    src = sessions[0]
    assert client.post("/api/record/start", json={"person": "T", "session": "one"}).status_code == 200
    n = int(40 * 50)
    for start in range(0, n, 50):
        batch = [[1000.0 + src.t[i], *src.x[i]] for i in range(start, start + 50)]
        client.post("/ingest", json={"samples": batch})
        if start == 0:
            assert client.post("/api/record/label", json={"label": "typing"}).status_code == 200
        server.engine.process()
    assert server.engine.current["activity"] is not None
    assert len(server.engine.timeline) > 20
    saved = client.post("/api/record/stop").json()
    assert 45 < saved["rate_hz"] < 55
    back = load_session(saved["path"])
    assert (back.cls == "typing").mean() > 0.9
