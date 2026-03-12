# WhisperSign Data Recorder

GUI application (PyQt5) for collecting and processing Vietnamese Sign Language hand gesture data using Leap Motion Controller. Outputs training-ready datasets compatible with the [WhisperSign](../Whisper_modification/) model.

## Features

- **Collect Tab**: Real-time hand skeleton visualization + gesture recording with countdown timer
- **Process Tab**: Scan raw recordings, view statistics, export train/val/test splits
- **Mock Mode**: Full functionality without Leap Motion hardware (random skeleton data)
- **Standalone**: No dependencies on other workspace modules

## Quick Start

```bash
pip install -r requirements.txt
python recorder/main.py
```

## Data Format

### Raw Recording (from Collect tab)
```
data/raw/{gloss_name}/{timestamp}.npy    # shape (T, 42, 7) float32
data/sessions.json                        # metadata index
```

Each frame has 42 joints (21 left + 21 right hand), 7 features per joint:
| Col | Feature    | Unit    |
|-----|------------|---------|
| 0-2 | x, y, z    | mm      |
| 3-5 | vx, vy, vz | mm/s    |
| 6   | confidence | 0.0-1.0 |

### Processed Output (from Process tab)
```
data/processed/
├── train/features/sample_00000.npy   # (T, 42, 7) float32
├── train/labels/sample_00000.npy     # (1,) int64
├── val/...
├── test/...
└── label_map.json                    # {"<blank>": 0, "gloss": 1, ...}
```

Compatible with `WhisperSign`'s `SignLanguageDataset` — load directly for training.

## Project Structure

```
recorder/
├── main.py              # Entry point
├── config.py            # Paths & constants
├── requirements.txt
├── core/
│   ├── session_manager.py   # Save/load .npy recordings + sessions.json
│   ├── data_processor.py    # Scan, validate, split, export pipeline
│   └── process_worker.py    # QThread wrapper for background processing
├── ui/
│   ├── main_window.py       # QMainWindow with 2 tabs
│   ├── collect_tab.py       # Recording UI + state machine
│   └── process_tab.py       # Processing UI + progress
└── utils/
    ├── lmc_capture.py        # LeapMotionCapture + MockLeapMotionCapture
    └── visualization.py      # HandVisualizer (skeleton renderer)
```

## Requirements

- Python 3.10+
- PyQt5, numpy, opencv-python
- (Optional) Leap Motion Controller + `leapc-python-api` for real hardware

## License

Part of the WhisperSign project.
