# Architecture — WhisperSign Data Recorder

Tài liệu giải thích chi tiết cách hoạt động của từng module trong recorder.

---

## Tổng quan Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    COLLECT TAB                                │
│                                                              │
│  LeapMotionCapture / MockLeapMotionCapture                   │
│  (background thread, 60fps)                                  │
│    │                                                         │
│    │  callback(timestamp_us, frame_np[42,7])                 │
│    ▼                                                         │
│  CollectTab._on_lmc_frame()  ← BACKGROUND THREAD            │
│    │  self._latest_frame = frame_np                          │
│    │  if RECORDING: buffer.append(frame_np.copy())           │
│    │  self.frame_received.emit()                             │
│    ▼                                                         │
│  QTimer (main thread, 30fps)                                 │
│    │  _render_tick() → HandVisualizer.render()               │
│    │  → BGR numpy → RGB → QImage → QPixmap → QLabel         │
│    ▼                                                         │
│  Live skeleton on screen                                     │
│                                                              │
│  Record → COUNTDOWN(3s) → RECORDING → Stop                  │
│    │  SessionManager.save_recording(gloss, frames)           │
│    │  np.stack(frames) → (T, 42, 7) float32                 │
│    ▼                                                         │
│  data/raw/{gloss}/{timestamp}.npy                            │
│  sessions.json updated                                       │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                    PROCESS TAB                                │
│                                                              │
│  DataProcessor.scan_raw_directory()                          │
│    → Count per-gloss, validate shapes, compute stats         │
│                                                              │
│  DataProcessor.process_and_export()                          │
│    │  1. Validate all .npy files                             │
│    │  2. Build label_map {"<blank>": 0, "gloss": 1, ...}    │
│    │  3. Stratified split (80/10/10 per gloss)               │
│    │  4. For each file:                                      │
│    │     - Load (T, 42, 7)                                   │
│    │     - Recompute velocity: np.gradient(pos, dt, axis=0)  │
│    │     - Save features + label .npy                        │
│    │  5. Save label_map.json                                 │
│    ▼                                                         │
│  data/processed/{train,val,test}/{features,labels}/*.npy     │
│  data/processed/label_map.json                               │
└──────────────────────────────────────────────────────────────┘
                          │
                          ▼
              WhisperSign SignLanguageDataset
              (SpatialNorm → ScaleNorm → pad → augment)
```

---

## Module chi tiết

### 1. `config.py` — Constants

```python
PROJECT_ROOT = Path(__file__).resolve().parent   # recorder/
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SESSIONS_FILE = DATA_DIR / "sessions.json"

DEFAULT_FPS = 60       # Leap Motion capture FPS
MOCK_FPS    = 60       # Mock capture FPS
VIS_FPS     = 30       # Skeleton render FPS (30fps đủ mượt, giảm tải CPU)
VIS_WIDTH   = 640
VIS_HEIGHT  = 480
COUNTDOWN_SECONDS = 3  # Đếm ngược trước khi ghi
```

Tại sao VIS_FPS = 30 mà capture = 60? Capture ở 60fps để data đủ dense cho training. Render ở 30fps vì mắt người chỉ cần ~30fps để thấy mượt, giảm tải main thread.

---

### 2. `utils/lmc_capture.py` — Hardware Abstraction

**2 classes chính:**

#### `LeapMotionCapture`
- Kết nối thật với Leap Motion Controller qua `leapc-python-api`
- Listener nhận tracking events trên background thread
- Mỗi frame: trích xuất 42 joints × 7 features từ SDK hand objects
- Gọi `on_frame_callback(timestamp_us, frame_np)` trên background thread

#### `MockLeapMotionCapture`
- Giả lập LMC cho development không cần hardware
- Background thread tạo random frames ở MOCK_FPS
- Frame shape giống thật: `(42, 7)` float32, confidence = 1.0
- Start/stop control giống LeapMotionCapture

#### `create_capture(on_frame_callback, mock, **kwargs)`
- Factory function: trả về `MockLeapMotionCapture` nếu `mock=True` hoặc nếu `leap` SDK không cài
- Auto-fallback: nếu không tìm thấy leap SDK → tự dùng mock + log warning

---

### 3. `utils/visualization.py` — HandVisualizer

Render hand skeleton lên numpy BGR canvas.

**Input**: `frame` shape `(42, 7)` — 42 joints, 7 features mỗi joint  
**Output**: BGR numpy array `(VIS_HEIGHT, VIS_WIDTH, 3)` uint8

**Xử lý:**
1. Chia frame thành 2 tay: `frame[:21]` (trái), `frame[21:]` (phải)
2. Project 3D → 2D: dùng tọa độ x, z (bird's-eye view) hoặc x, y (frontal)
3. Vẽ connections theo MediaPipe hand topology (wrist → MCP → PIP → DIP → TIP)
4. Joint size tỷ lệ với confidence
5. Mỗi tay một màu khác nhau (xanh/cam)

---

### 4. `core/session_manager.py` — Recording Storage

**Chức năng:** Lưu recording thành `.npy` file + tracking metadata trong `sessions.json`.

#### `save_recording(gloss, frames) → Path`
1. `np.stack(frames)` → shape `(T, 42, 7)` float32
2. Validate shape: phải là `(T, 42, 7)`
3. Tạo thư mục `data/raw/{gloss}/`
4. Filename: `{YYYYMMDD_HHMMSS_ffffff}.npy` (microsecond precision tránh trùng)
5. `np.save(filepath, sequence)`
6. Append entry vào `sessions.json`:
   ```json
   {
     "gloss": "xin_chao",
     "file": "xin_chao/20260312_143000_123456.npy",
     "num_frames": 180,
     "shape": [180, 42, 7],
     "timestamp": "2026-03-12T14:30:00"
   }
   ```

#### `get_gloss_counts() → dict`
Đếm recordings per gloss từ sessions.json: `{"xin_chao": 5, "cam_on": 3}`

#### `get_all_glosses() → list`
Sorted list tên gloss đã ghi.

---

### 5. `ui/collect_tab.py` — Recording UI

**State Machine:**
```
       ┌─────── Cancel ──────────┐
       │                         │
       ▼                         │
    ┌──────┐  Record  ┌──────────┴──┐  3s done  ┌───────────┐
    │ IDLE │────────→ │  COUNTDOWN  │──────────→│ RECORDING │
    └──────┘          └─────────────┘           └─────┬─────┘
       ▲                                              │
       │             ┌────────┐          Stop         │
       └─────────────│ SAVING │◄──────────────────────┘
                     └────────┘
```

**Trạng thái → UI:**
| State     | Record btn | Stop btn | Cancel btn | Gloss input |
|-----------|-----------|----------|------------|-------------|
| IDLE      | ✅        | ❌       | ❌         | ✅          |
| COUNTDOWN | ❌        | ❌       | ✅         | ❌          |
| RECORDING | ❌        | ✅       | ✅         | ❌          |
| SAVING    | ❌        | ❌       | ❌         | ❌          |

**Threading model:**
- `MockLeapMotionCapture` gọi `_on_lmc_frame()` trên **background thread**
- `_on_lmc_frame()` chỉ lưu reference + buffer + emit signal (thread-safe nhờ Python GIL)
- `QTimer` ở 30fps gọi `_render_tick()` trên **main thread** → an toàn update UI
- Signal `frame_received` tự marshal sang main thread bởi Qt

**Tại sao dùng `frame_np.copy()` khi buffer?**
Mock/real capture có thể reuse cùng numpy array cho frame tiếp theo. Nếu không copy, tất cả frames trong buffer trỏ tới cùng vùng nhớ → dữ liệu bị ghi đè.

**Mock toggle:**
- Checkbox "Mock Mode" cho phép bật/tắt runtime
- Khi toggle: stop capture → tạo capture mới → start lại
- Bị block nếu đang recording (tránh mất data)

---

### 6. `core/data_processor.py` — Processing Pipeline

#### `scan_raw_directory(raw_dir) → list[dict]`
- Duyệt tất cả subfolder trong `raw_dir/`
- Mỗi subfolder = 1 gloss
- Mỗi `.npy` file: validate shape `(T, 42, 7)` bằng `mmap_mode='r'` (không load vào RAM)
- Trả về stats per gloss: count, valid, invalid, avg/min/max frames

#### `estimate_split(stats) → dict`
- Tính estimate 80/10/10 split
- Trả về `{total, train, val, test, glosses}`

#### `process_and_export(raw_dir, output_dir, ...) → dict`

**Pipeline chi tiết:**

1. **Scan** — Thu thập tất cả file `.npy` hợp lệ, nhóm theo gloss
2. **Label map** — `{"<blank>": 0, "cam_on": 1, "xin_chao": 2, ...}`
   - `<blank>` = 0 bắt buộc cho CTC loss (Connectionist Temporal Classification)
   - Glosses sorted alphabetically, đánh số từ 1
3. **Stratified split** — Với mỗi gloss riêng:
   - Shuffle files (seed=42 cho reproducible)
   - 80% train, 10% val, 10% test
   - Nếu gloss ≤ 2 files → tất cả vào train (không đủ để split)
4. **Velocity recompute** — Với mỗi file:
   ```python
   positions = data[:, :, :3]              # (T, 42, 3)
   velocities = np.gradient(positions, dt, axis=0)  # dt = 1/60
   data[:, :, 3:6] = velocities
   ```
   Tại sao recompute? Vì mock data có velocity random, và real LMC velocity dùng palm velocity (không chính xác cho per-joint). `np.gradient` tính central difference chính xác hơn.
5. **Export** — Mỗi sample thành 2 file:
   - `features/sample_XXXXX.npy` → `(T, 42, 7)` float32 (T variable, NOT padded)
   - `labels/sample_XXXXX.npy` → `(1,)` int64 (gloss ID)
6. **label_map.json** — Lưu mapping gloss ↔ ID

**Tại sao KHÔNG normalize ở đây?**
`SignLanguageDataset.__getitem__()` sẽ apply `SpatialNorm` + `ScaleNorm` + padding lúc training. Normalize 2 lần = data sai.

---

### 7. `core/process_worker.py` — Background Thread

QThread wrapper cho `DataProcessor.process_and_export()` để không block UI.

**Signals:**
- `progress(int, str)` — percent + message → update progress bar + log
- `finished(dict)` — result dict → show summary  
- `error(str)` — error message → show dialog

---

### 8. `ui/process_tab.py` — Processing UI

**Layout:**
```
┌─────────────────────────────────────────────┐
│ Input Directory: [path............] Browse Scan│
├─────────────────────────────────────────────┤
│ Gloss | Count | Valid | Avg Frames | ...    │
│ ──────┼───────┼───────┼────────────┼─────── │
│ cam_on│   3   │   3   │    180     │ ...    │
│ xin.. │   5   │   5   │    200     │ ...    │
├─────────────────────────────────────────────┤
│ Summary: 8 samples, 2 glosses. Est: 6/1/1  │
├─────────────────────────────────────────────┤
│ [▶ Process & Export]                         │
│ [████████████████░░░░░░░░░░] 60%            │
│ [train] Exported 3/6                         │
│ [val] Exported 1/1                           │
└─────────────────────────────────────────────┘
```

---

### 9. `ui/main_window.py` — App Shell

- `QMainWindow` với `QTabWidget` chứa 2 tab
- Auto-start capture khi mở
- `closeEvent` → stop capture (cleanup threads)

---

## Joint Format: (42, 7)

42 joints = 21 trái + 21 phải. Mapping theo MediaPipe Hand Landmarks:

```
Mỗi tay (21 joints):
  0  = Wrist
  1  = Thumb CMC      5  = Index MCP      9  = Middle MCP    13 = Ring MCP     17 = Pinky MCP
  2  = Thumb MCP      6  = Index PIP     10  = Middle PIP    14 = Ring PIP     18 = Pinky PIP
  3  = Thumb IP       7  = Index DIP     11  = Middle DIP    15 = Ring DIP     19 = Pinky DIP
  4  = Thumb TIP      8  = Index TIP     12  = Middle TIP    16 = Ring TIP     20 = Pinky TIP

Left hand  → indices [0, 20]
Right hand → indices [21, 41]
```

7 features per joint:
```
[x, y, z, vx, vy, vz, confidence]
 mm  mm  mm  mm/s mm/s mm/s  0-1
```

---

## Tích hợp với WhisperSign Training

```
recorder/data/processed/    ←── output từ Process tab
        │
        ▼
SignLanguageDataset(data_dir="recorder/data/processed", split="train")
        │
        │ __getitem__(idx):
        │   1. Load features (T, 42, 7) float32
        │   2. SpatialNorm: center around wrist, align hands
        │   3. ScaleNorm: normalize to unit scale
        │   4. Pad/truncate to 1500 frames
        │   5. Augment: GestureMasking, TemporalJitter, NoiseInjection
        │
        ▼
    Tensor (1500, 42, 7)  →  WhisperSign Model
```
