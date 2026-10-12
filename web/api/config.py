"""Web layer configuration — paths and constants."""
from __future__ import annotations

import sys
from pathlib import Path

# Project root (PPG-Agent/), so `import ppg_agent` works regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Heavy derived DB lives on the data disk — single source of truth in
# ppg_agent.config (PPG_DB_PATH / DATA_ROOT/ppg_agent/ resolution)
from ppg_agent.config import PHENOTYPE_DB
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
DEMO_CASES_FILE = PROJECT_ROOT / "web" / "demo_cases" / "cases.jsonl"

# Waveform display
DEFAULT_DISPLAY_POINTS = 2000
MAX_DISPLAY_POINTS = 5000

# Signal convention used across the project (see analysis scripts / signal_engine)
TARGET_FS = 500

API_PREFIX = "/api"
