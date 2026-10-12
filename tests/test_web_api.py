"""Smoke tests for the web API: data browsing + segment waveform endpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from web.api.main import app


def test_health_and_datasets():
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        datasets = client.get("/api/data/datasets").json()
        sources = {d["source"] for d in datasets}
        assert {"vital", "mimic", "aurora"} <= sources
        aurora = next(d for d in datasets if d["source"] == "aurora")
        assert aurora["scenarios"].get("a", 0) > 0  # scenario split present


def test_subjects_filtering():
    with TestClient(app) as client:
        r = client.get("/api/data/subjects",
                       params={"source": "aurora", "scenario": "a", "page_size": 5})
        body = r.json()
        assert body["total"] > 0
        for item in body["items"]:
            assert item["source"] == "aurora" and item["scenario"] == "a"


def test_subject_detail_and_waveform():
    with TestClient(app) as client:
        r = client.get("/api/data/subjects",
                       params={"source": "aurora", "scenario": "a", "page_size": 1})
        subj = r.json()["items"][0]
        detail = client.get(
            f"/api/data/subjects/{subj['source']}/{subj['subject_id']}"
        ).json()
        assert detail["segments"], "subject must have segments"
        seg_id = detail["segments"][0]["id"]

        wf = client.get(f"/api/data/segments/{seg_id}/waveform",
                        params={"points": 500}).json()
        assert len(wf["points"]) == 500
        assert wf["fs"] == 500 and wf["duration_sec"] > 0
        assert wf["stats"]["max"] >= wf["stats"]["min"]


def test_segment_waveform_endpoint():
    """GET /api/analysis/segment-waveform 冒烟：对话页历史段 Tab 懒加载端点。

    Contract: returns normalized [0,1] points on the fixed 10 s axis plus a
    beat overlay (absolute 500 Hz keypoints), matching the display-downsample
    vs overlay-fs alignment WaveformChart relies on.  Unknown segment → 404.
    """
    with TestClient(app) as client:
        subj = client.get("/api/data/subjects",
                          params={"source": "vital", "page_size": 1}).json()["items"][0]
        sid = subj["subject_id"]
        detail = client.get(f"/api/data/subjects/vital/{sid}").json()
        seg_idxs = [s["seg_idx"] for s in detail.get("segments", [])]
        assert seg_idxs, "subject must expose segments"

        ok = None
        for idx in seg_idxs[:5]:  # first readable segment (some files may be absent)
            r = client.get("/api/analysis/segment-waveform",
                           params={"source": "vital", "subject_id": sid,
                                   "seg_idx": idx, "points": 1000})
            if r.status_code == 200:
                ok = r.json()
                break
        assert ok is not None, "at least one segment must load"
        assert len(ok["points"]) == 1000
        assert ok["fs"] > 0 and ok["duration_sec"] > 0
        assert min(ok["points"]) >= -0.05 and max(ok["points"]) <= 1.05  # min-max [0,1]
        assert ok["overlay"]["fs"] > 0 and isinstance(ok["overlay"]["beats"], list)

        # missing segment → 404 (the lazy-fetch endpoint's error path)
        r404 = client.get("/api/analysis/segment-waveform",
                          params={"source": "vital", "subject_id": "__no_such__",
                                  "seg_idx": 0})
        assert r404.status_code == 404


if __name__ == "__main__":
    test_health_and_datasets()
    test_subjects_filtering()
    test_subject_detail_and_waveform()
    test_segment_waveform_endpoint()
    print("All web API smoke tests passed.")
