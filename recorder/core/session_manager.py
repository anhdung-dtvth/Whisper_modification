import json
from pathlib import Path
import numpy as np
from datetime import datetime
from recorder.config import RAW_DIR, SESSIONS_FILE


class SessionManager():
    def __init__(self, raw_dir: Path = RAW_DIR, session_file: Path = SESSIONS_FILE):
        self._raw_dir = raw_dir
        self._session_file = session_file
        self._sessions = self._load_sessions()

    def _load_sessions(self) -> dict:
        if self._session_file.exists():
            try:
                with open(self._session_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and isinstance(data.get("recordings", None), list):
                    return data
            except (json.JSONDecodeError, OSError):
                pass
        return {"recordings": []}
    def _save_sessions(self):
        self._session_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._session_file, "w") as f:
            json.dump(self._sessions, f, indent=2, ensure_ascii=False)
            
    def save_recording(self, gloss: str, frames: list[np.ndarray]):
        """     
        Lưu 1 recording.
        Args:
            gloss: Tên gesture (ví dụ "xin_chao")
            frames: List of numpy arrays, mỗi cái shape (42, 7)

        Returns:
            Path tới file .npy đã lưu
        """
        if not isinstance(gloss, str) or not gloss.strip():
            raise ValueError("gloss must be a non-empty string")
        if not frames:
            raise ValueError("frames must not be empty")

        gloss = gloss.strip()

        sequence = np.stack(frames, axis=0).astype(np.float32, copy=False)
        if sequence.ndim != 3 or sequence.shape[1:] != (42, 7):
            raise ValueError(f"Expected shape (T, 42, 7), got {sequence.shape}")

        gloss_dir = self._raw_dir / gloss
        gloss_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        filename = now.strftime("%Y%m%d_%H%M%S_%f") + ".npy"
        filepath = gloss_dir / filename

        np.save(filepath, sequence)

        entry = {
            "gloss": gloss,
            "file": str(filepath.relative_to(self._raw_dir)),
            "num_frames": int(sequence.shape[0]),
            "shape": list(sequence.shape),
            "timestamp": now.isoformat(timespec="seconds"),
        }
        self._sessions.setdefault("recordings", []).append(entry)
        self._save_sessions()

        return filepath

    def get_gloss_counts(self) -> dict:
            """
            Đếm số recordings cho mỗi gloss. Return: {"xin_chao": 5, "cam_on": 3}
            """
            counts: dict[str, int] = {}
            for recording in self._sessions.get("recordings", []):
                gloss = recording.get("gloss")
                if not gloss:
                    continue
                counts[gloss] = counts.get(gloss, 0) + 1
            return counts

    def get_all_glosses(self) -> list:
            """
            Return danh sách tên gloss đã ghi (sorted).
            """
            return sorted(self.get_gloss_counts().keys())