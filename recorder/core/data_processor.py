import json
import random
import numpy as np
from pathlib import Path
from typing import Callable, Optional


class DataProcessor:

    def scan_raw_directory(self, raw_dir: Path) -> list:
        """Scan raw directory and return per-gloss statistics."""
        stats = []
        if not raw_dir.exists():
            return stats

        for gloss_dir in sorted(raw_dir.iterdir()):
            if not gloss_dir.is_dir():
                continue

            npy_files = sorted(gloss_dir.glob("*.npy"))
            if not npy_files:
                continue

            valid = 0
            invalid = 0
            frame_counts = []

            for f in npy_files:
                try:
                    data = np.load(f, mmap_mode="r")
                    if data.ndim == 3 and data.shape[1] == 42 and data.shape[2] == 7:
                        valid += 1
                        frame_counts.append(data.shape[0])
                    else:
                        invalid += 1
                except Exception:
                    invalid += 1

            if frame_counts:
                avg_frames = int(np.mean(frame_counts))
                min_frames = int(np.min(frame_counts))
                max_frames = int(np.max(frame_counts))
            else:
                avg_frames = min_frames = max_frames = 0

            stats.append({
                "gloss": gloss_dir.name,
                "count": len(npy_files),
                "valid": valid,
                "invalid": invalid,
                "avg_frames": avg_frames,
                "avg_duration_s": round(avg_frames / 60, 2),
                "min_frames": min_frames,
                "max_frames": max_frames,
            })

        return stats

    def estimate_split(self, stats: list) -> dict:
        """Estimate train/val/test split counts (80/10/10)."""
        total = sum(s["valid"] for s in stats)
        n_test = max(1, round(total * 0.1)) if total > 0 else 0
        n_val = max(1, round(total * 0.1)) if total > 0 else 0
        n_train = max(0, total - n_test - n_val)
        return {
            "total": total,
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "glosses": len(stats),
        }

    def process_and_export(
        self,
        raw_dir: Path,
        output_dir: Path,
        target_fps: int = 60,
        test_ratio: float = 0.1,
        val_ratio: float = 0.1,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> dict:
        """
        Full pipeline: raw .npy -> processed train/val/test splits.

        Steps:
        1. Scan raw directory for valid .npy files
        2. Build label_map (CTC blank=0, glosses from 1)
        3. Recompute velocities from positions
        4. Stratified split per gloss
        5. Export features/*.npy + labels/*.npy per split
        6. Save label_map.json
        """
        def _emit(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        _emit(0, "Scanning raw directory...")

        # 1. Collect all valid files grouped by gloss
        gloss_files: dict[str, list[Path]] = {}
        for gloss_dir in sorted(raw_dir.iterdir()):
            if not gloss_dir.is_dir():
                continue
            files = []
            for f in sorted(gloss_dir.glob("*.npy")):
                try:
                    data = np.load(f, mmap_mode="r")
                    if data.ndim == 3 and data.shape[1] == 42 and data.shape[2] == 7:
                        files.append(f)
                except Exception:
                    pass
            if files:
                gloss_files[gloss_dir.name] = files

        if not gloss_files:
            raise ValueError(f"No valid .npy files found in {raw_dir}")

        _emit(10, f"Found {sum(len(v) for v in gloss_files.values())} samples "
              f"across {len(gloss_files)} glosses.")

        # 2. Build label_map
        label_map = {"<blank>": 0}
        for i, gloss in enumerate(sorted(gloss_files.keys()), start=1):
            label_map[gloss] = i

        # 3+4. Stratified split
        rng = random.Random(42)
        split_assignments: dict[str, list[tuple[Path, int]]] = {
            "train": [], "val": [], "test": [],
        }

        for gloss, files in gloss_files.items():
            label_id = label_map[gloss]
            shuffled = list(files)
            rng.shuffle(shuffled)
            n = len(shuffled)

            if n <= 2:
                # Too few samples — put all in train
                for f in shuffled:
                    split_assignments["train"].append((f, label_id))
            else:
                n_test = max(1, round(n * test_ratio))
                n_val = max(1, round(n * val_ratio))
                n_train = n - n_test - n_val
                if n_train < 1:
                    n_train = 1
                    n_val = max(0, n - n_train - n_test)

                for f in shuffled[:n_train]:
                    split_assignments["train"].append((f, label_id))
                for f in shuffled[n_train:n_train + n_val]:
                    split_assignments["val"].append((f, label_id))
                for f in shuffled[n_train + n_val:]:
                    split_assignments["test"].append((f, label_id))

        _emit(30, "Split done. Processing and exporting...")

        # 5. Export
        dt = 1.0 / target_fps
        result_counts = {}
        total_files = sum(len(v) for v in split_assignments.values())
        processed = 0

        for split_name, items in split_assignments.items():
            feat_dir = output_dir / split_name / "features"
            lab_dir = output_dir / split_name / "labels"
            feat_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)

            for idx, (filepath, label_id) in enumerate(items):
                data = np.load(filepath).astype(np.float32, copy=False)

                # Recompute velocities from positions
                positions = data[:, :, :3]  # (T, 42, 3)
                velocities = np.gradient(positions, dt, axis=0)  # (T, 42, 3)
                data[:, :, 3:6] = velocities

                sample_name = f"sample_{idx:05d}.npy"
                np.save(feat_dir / sample_name, data)
                np.save(lab_dir / sample_name, np.array([label_id], dtype=np.int64))

                processed += 1
                pct = 30 + int(60 * processed / total_files)
                if processed % 10 == 0 or processed == total_files:
                    _emit(pct, f"[{split_name}] Exported {idx + 1}/{len(items)}")

            result_counts[split_name] = len(items)

        # 6. Save label_map.json
        label_map_path = output_dir / "label_map.json"
        with open(label_map_path, "w") as f:
            json.dump(label_map, f, indent=2, ensure_ascii=False)

        _emit(95, "Saved label_map.json")
        _emit(100, "Done!")

        return {
            "train_count": result_counts.get("train", 0),
            "val_count": result_counts.get("val", 0),
            "test_count": result_counts.get("test", 0),
            "num_glosses": len(gloss_files),
            "label_map": label_map,
        }
