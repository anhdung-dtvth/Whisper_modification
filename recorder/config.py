"""
Nội dung cần có:
- PROJECT_ROOT = Path(__file__).resolve().parent         # → .../recorder/
- DATA_DIR = PROJECT_ROOT / "data"
- RAW_DIR = DATA_DIR / "raw"
- PROCESSED_DIR = DATA_DIR / "processed"
- SESSIONS_FILE = DATA_DIR / "sessions.json"
- DEFAULT_FPS = 60
- MOCK_FPS = 60
- VIS_FPS = 30      # Hz cập nhật skeleton trên UI
- VIS_WIDTH = 640
- VIS_HEIGHT = 480
- COUNTDOWN_SECONDS = 3
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SESSIONS_FILE = DATA_DIR / "sessions.json"
DEFAULT_FPS = 60
MOCK_FPS = 60
VIS_FPS = 30      # Hz cập nhật skeleton trên UI
VIS_WIDTH = 640
VIS_HEIGHT = 480
COUNTDOWN_SECONDS = 3

