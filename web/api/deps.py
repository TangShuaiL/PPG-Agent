"""Database dependencies and helpers."""
from __future__ import annotations

import sqlite3
from typing import Iterator

from ppg_agent.database.connection import PHENOTYPE_DB, phenotype_conn

from . import config


def get_phenotype_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency yielding a phenotype DB connection."""
    conn = phenotype_conn()
    try:
        yield conn
    finally:
        conn.close()


def enable_wal_once() -> None:
    """Ensure WAL mode is persisted on the phenotype DB (called at startup)."""
    for path in (config.PHENOTYPE_DB,):
        if path.exists():
            conn = sqlite3.connect(str(path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.close()


def resolve_segment(seg_id: int) -> dict:
    """Look up (source, subject_id, seg_idx) for a segment id.

    Shared by the chat router's tool handlers (formerly in tasks.runner,
    retired with the async task subsystem).
    """
    with phenotype_conn() as conn:
        row = conn.execute(
            "SELECT source, subject_id, seg_idx FROM segments WHERE id=?",
            (seg_id,),
        ).fetchone()
    if row is None:
        raise ValueError(f"segment {seg_id} not found")
    return dict(row)
