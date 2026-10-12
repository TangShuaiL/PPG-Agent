"""Phenotype-database connections (read-mostly).

Single home for DB access used by both the agent layer (dataset-adaptation
subject pools) and the web layer (data browsing, analysis).  The web deps
module re-exports ``phenotype_conn`` from here so the connection behaviour
stays identical for every caller.
"""
from __future__ import annotations

import sqlite3

from ppg_agent.config import PHENOTYPE_DB


def _connect(path) -> sqlite3.Connection:
    # check_same_thread=False: under Python 3.13 + anyio the FastAPI sync
    # dependency and the sync endpoint can land on different worker threads.
    # The connection is request-scoped (created → used → closed serially),
    # so cross-thread *reuse* is impossible; only the thread identity differs.
    conn = sqlite3.connect(str(path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL allows concurrent readers while an analysis task writes elsewhere
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def phenotype_conn() -> sqlite3.Connection:
    """Read-mostly connection to the phenotype database."""
    conn = _connect(PHENOTYPE_DB)
    conn.execute("PRAGMA query_only=1")
    return conn
