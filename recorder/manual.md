# User Manual — WhisperSign Data Recorder

## Installation

```bash
cd recorder
pip install -r requirements.txt
```

For Leap Motion hardware support, also install `leapc-python-api` from the `leapc-python-bindings/` directory.

## Launch

```bash
python recorder/main.py
```

The application opens with two tabs: **Collect** and **Process**.

---

## Tab 1: Collect — Recording Gestures

### Interface

| Area | Description |
|------|-------------|
| Left panel (60%) | Live hand skeleton visualization (640×480, 30fps) |
| Right panel (40%) | Recording controls |

### Controls

- **Mock Mode checkbox**: Toggle between mock data (random skeleton) and real Leap Motion hardware. Cannot switch while recording.
- **Gloss Name**: Type a new gesture name or select from previously recorded ones (dropdown).
- **🔴 Record**: Start a recording. Triggers a 3-second countdown.
- **⏹ Stop & Save**: Stop recording and save the captured frames as `.npy`.
- **❌ Cancel**: Abort current countdown or recording without saving.

### Recording Workflow

1. Enter or select a gloss name (e.g., `xin_chao`, `cam_on`)
2. Click **Record**
3. Wait for countdown: **3 → 2 → 1**
4. Perform the gesture while **● REC** is displayed
5. Click **Stop & Save** when finished
6. The Statistics panel updates with the new count

### Recording State Machine

```
IDLE → Record → COUNTDOWN (3s) → RECORDING → Stop → SAVING → IDLE
                                            → Cancel → IDLE
```

Buttons are enabled/disabled automatically based on the current state.

### Output Files

Each recording saves to:
```
recorder/data/raw/{gloss_name}/{YYYYMMDD_HHMMSS_ffffff}.npy
```
- Shape: `(T, 42, 7)` where T = number of captured frames
- dtype: `float32`
- Metadata appended to `recorder/data/sessions.json`

---

## Tab 2: Process — Export Training Data

### Interface

| Area | Description |
|------|-------------|
| Top bar | Directory selector + Scan button |
| Table | Per-gloss statistics (count, valid, avg frames, duration, min, max) |
| Summary | Total samples, glosses, estimated split |
| Process button | Run the export pipeline |
| Progress bar | Export progress |
| Log area | Detailed processing messages |

### Workflow

1. The input directory defaults to `recorder/data/raw/`
2. Click **🔍 Scan** to analyze all recordings
3. Review the statistics table and summary
4. Click **▶ Process & Export** to run the pipeline:
   - Validates all `.npy` files (shape must be `(T, 42, 7)`)
   - Recomputes velocities from positions using `np.gradient`
   - Builds label map (`<blank>=0`, glosses from 1)
   - Stratified 80/10/10 train/val/test split per gloss
   - Exports to `recorder/data/processed/`

### Output Structure

```
recorder/data/processed/
├── train/
│   ├── features/sample_00000.npy   # (T, 42, 7) float32
│   └── labels/sample_00000.npy     # (1,) int64 — gloss ID
├── val/
│   ├── features/...
│   └── labels/...
├── test/
│   ├── features/...
│   └── labels/...
└── label_map.json
```

### label_map.json

```json
{
  "<blank>": 0,
  "cam_on": 1,
  "xin_chao": 2
}
```

`<blank>` at index 0 is required by the CTC loss function used in WhisperSign training.

---

## Using with WhisperSign Training

After exporting, load the processed data directly:

```python
from Whisper_modification.src.data.dataset import SignLanguageDataset

dataset = SignLanguageDataset(
    data_dir="recorder/data/processed",
    split="train",
)
sample = dataset[0]
print(sample["features"].shape)  # torch.Size([1500, 42, 7])
print(sample["labels"].shape)    # torch.Size([1])
```

The `SignLanguageDataset` applies `SpatialNorm`, `ScaleNorm`, padding to 1500 frames, and augmentations automatically.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Window doesn't open | Check `pip install PyQt5` |
| No skeleton visible | Ensure Mock Mode is checked (or Leap Motion is connected) |
| "No valid .npy files" on Process | Record some gestures first in Collect tab |
| Gloss with <3 samples skips val/test | Record at least 3 samples per gloss for proper splitting |
