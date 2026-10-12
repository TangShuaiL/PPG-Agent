"""Data browsing: datasets / subjects / segments / waveforms (+ upload)."""
from __future__ import annotations

import shutil
import sqlite3
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile

from .. import config
from ..deps import get_phenotype_db
from ..services.waveform_reader import get_waveform_payload

router = APIRouter(prefix="/data", tags=["data"])


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    source: str = Form("custom"),
):
    """Save an uploaded raw waveform file for later ingestion.

    Files land in data/uploads/<source>/<uuid>_<name>; database ingestion
    (segment registration / feature extraction) is a follow-up pipeline.
    """
    if not file.filename or "." not in file.filename:
        raise HTTPException(400, "file must have an extension")
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in {"mat", "h5", "csv", "txt"}:
        raise HTTPException(400, f"unsupported extension .{ext} (mat/h5/csv/txt)")
    target_dir = config.UPLOAD_DIR / source
    target_dir.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = target_dir / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"filename": file.filename, "stored": str(dest.relative_to(config.UPLOAD_DIR)),
            "size_bytes": dest.stat().st_size, "uploaded_at": time.time()}


@router.get("/datasets")
def list_datasets(db: sqlite3.Connection = Depends(get_phenotype_db)):
    """Per-dataset overview: subject/segment counts, quality mix, scenarios."""
    subjects = db.execute(
        "SELECT source, scenario, COUNT(*) AS n FROM subjects"
        " GROUP BY source, scenario"
    ).fetchall()
    segments = db.execute(
        "SELECT source, quality_tier, COUNT(*) AS n FROM segments"
        " GROUP BY source, quality_tier"
    ).fetchall()
    out: dict[str, dict] = {}
    for r in subjects:
        d = out.setdefault(r["source"], {"source": r["source"], "scenarios": {}, "subjects": 0})
        d["subjects"] += r["n"]
        if r["scenario"]:
            d["scenarios"][r["scenario"]] = r["n"]
    for r in segments:
        d = out.setdefault(r["source"], {"source": r["source"], "scenarios": {}, "subjects": 0})
        d.setdefault("segments", {})[r["quality_tier"]] = r["n"]
    return list(out.values())


@router.get("/subjects")
def list_subjects(
    source: str | None = Query(default=None),
    scenario: str | None = Query(default=None, description="aurora only: a=lab, o=dynamic"),
    gender: str | None = Query(default=None),
    age_min: float | None = Query(default=None),
    age_max: float | None = Query(default=None),
    split: str | None = Query(default=None, description="train|test"),
    q: str | None = Query(default=None, description="subject_id substring filter"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    db: sqlite3.Connection = Depends(get_phenotype_db),
):
    where, args = ["1=1"], []
    for col, val in [("source", source), ("scenario", scenario), ("gender", gender),
                     ("split", split)]:
        if val:
            where.append(f"{col}=?")
            args.append(val)
    if q:
        where.append("subject_id LIKE ?")
        args.append(f"%{q}%")
    if age_min is not None:
        where.append("age>=?")
        args.append(age_min)
    if age_max is not None:
        where.append("age<=?")
        args.append(age_max)
    cond = " AND ".join(where)

    total = db.execute(f"SELECT COUNT(*) FROM subjects WHERE {cond}", args).fetchone()[0]
    rows = db.execute(
        f"SELECT subject_id, source, scenario, age, gender, n_segments, split"
        f" FROM subjects WHERE {cond} ORDER BY source, subject_id"
        f" LIMIT ? OFFSET ?",
        args + [page_size, (page - 1) * page_size],
    ).fetchall()
    return {"total": total, "page": page, "page_size": page_size,
            "items": [dict(r) for r in rows]}


@router.get("/subjects/{source}/{subject_id}")
def get_subject(source: str, subject_id: str,
                db: sqlite3.Connection = Depends(get_phenotype_db)):
    subj = db.execute(
        "SELECT * FROM subjects WHERE source=? AND subject_id=?",
        (source, subject_id),
    ).fetchone()
    if subj is None:
        raise HTTPException(404, f"subject {source}/{subject_id} not found")
    segs = db.execute(
        "SELECT id, seg_idx, quality_score, quality_tier, n_beats_valid,"
        " n_beats_total, sbp_label, dbp_label"
        " FROM segments WHERE source=? AND subject_id=? ORDER BY seg_idx"
        " LIMIT 500",
        (source, subject_id),
    ).fetchall()
    return {"subject": dict(subj), "segments": [dict(s) for s in segs]}


@router.get("/segments/{seg_id}")
def get_segment(seg_id: int, db: sqlite3.Connection = Depends(get_phenotype_db)):
    seg = db.execute("SELECT * FROM segments WHERE id=?", (seg_id,)).fetchone()
    if seg is None:
        raise HTTPException(404, f"segment {seg_id} not found")
    return dict(seg)


@router.get("/segments/{seg_id}/waveform")
def get_segment_waveform(
    seg_id: int,
    points: int = Query(default=2000, ge=200, le=5000),
    db: sqlite3.Connection = Depends(get_phenotype_db),
):
    seg = db.execute(
        "SELECT id, source, subject_id, seg_idx, waveform_path FROM segments WHERE id=?",
        (seg_id,),
    ).fetchone()
    if seg is None:
        raise HTTPException(404, f"segment {seg_id} not found")
    try:
        payload = get_waveform_payload(seg["waveform_path"], seg["seg_idx"], points)
    except FileNotFoundError:
        raise HTTPException(404, f"waveform file missing: {seg['waveform_path']}")
    except Exception as e:  # h5py / scipy failures
        raise HTTPException(500, f"failed to load waveform: {e}")
    payload["segment"] = dict(seg)
    return payload
