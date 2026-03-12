# Plan: GUI Thu Thập & Xử Lý Data cho WhisperSign (Chi Tiết)

## TL;DR
App standalone PyQt5 trong `recorder/`, 2 tab: **Collect** (ghi gesture từ Leap Motion) và **Process** (xử lý + export cho training). Reuse `HandVisualizer`, `lmc_capture.py`, `SpatialNorm`, `ScaleNorm`. Mock mode cho dev không cần phần cứng.

**Focus học**: PyQt5 từ zero (signal-slot, QThread, layout) + Numpy data I/O.

---

## Đã hoàn thành
- [x] Phase 0: Sửa bugs leap_motion_extract.py (4 lỗi)

---

## Phase 1: Cấu trúc project + Entry point

### Mục tiêu
Chạy được cửa sổ trống với 2 tab, hiểu cách tổ chức PyQt5 app.

### Khái niệm PyQt5 cần hiểu trước

**QApplication** — "Bộ não" của mọi PyQt5 app. Mỗi app chỉ có DUY NHẤT 1 instance. Nó quản lý event loop (vòng lặp chờ + xử lý sự kiện: click, hover, timer, repaint...).

**QMainWindow** — Cửa sổ chính, có sẵn: menu bar, toolbar, status bar, central widget. Bạn đặt nội dung chính vào central widget.

**QTabWidget** — Widget chứa nhiều tab, mỗi tab là 1 QWidget riêng. Gọi `addTab(widget, "Tên Tab")` để thêm.

**Signal-Slot** — Cơ chế giao tiếp giữa các object trong Qt:
- **Signal**: "Sự kiện đã xảy ra" (ví dụ: button clicked, data sẵn sàng)
- **Slot**: "Hàm sẽ chạy khi signal phát" (ví dụ: update UI)
- Kết nối: `button.clicked.connect(self.on_button_click)`
- Có thể kết nối 1 signal → nhiều slot, hoặc nhiều signal → 1 slot

### Files tạo mới

#### 1.1 `recorder/__init__.py`
File trống, đánh dấu đây là Python package.

#### 1.2 `recorder/config.py`
Chứa constants và paths. Không cần YAML phức tạp ở giai đoạn này.

```
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
```

**Tips**: Dùng `pathlib.Path` thay vì `os.path.join` — code sạch hơn, cross-platform.

#### 1.3 `recorder/main.py` — Entry point

**Pattern tham khảo**: `whisper_app/main.py` dòng 46–91

```
Flow:
1. Thêm project root vào sys.path (để import được whisper_app.*, Whisper_modification.*)
2. Tạo QApplication(sys.argv)
3. Đặt style "Fusion" (giao diện nhất quán trên mọi OS)
4. Tạo MainWindow()
5. window.show()
6. sys.exit(app.exec_())   ← BẮT ĐẦU EVENT LOOP (app chạy ở đây cho tới khi đóng cửa sổ)
```

**Code hint — sys.path setup**:
```python
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
# parent.parent vì: recorder/main.py → recorder/ → WhisperLMCVSL/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
```
Sau dòng này bạn có thể `from whisper_app.app.ui.visualization import HandVisualizer`.

**Lưu ý**: `app.exec_()` BLOCK — code sau dòng này chỉ chạy khi user đóng cửa sổ.

#### 1.4 `recorder/ui/__init__.py`
File trống.

#### 1.5 `recorder/ui/main_window.py` — Cửa sổ chính

**Pattern tham khảo**: `whisper_app/app/ui/main_window.py`

```
Class: RecorderWindow(QMainWindow)
  __init__(self, parent=None):
    1. super().__init__(parent)
    2. self.setWindowTitle("WhisperSign Data Recorder")
    3. self.resize(1000, 700)

    4. Tạo QTabWidget
    5. Tạo 2 placeholder QWidget (sẽ thay bằng tab thật ở Phase 2, 4)
    6. tab_widget.addTab(collect_placeholder, "📹 Collect")
       tab_widget.addTab(process_placeholder, "⚙️ Process")
    7. self.setCentralWidget(tab_widget)

    8. (Tùy chọn) Thêm QStatusBar: self.statusBar().showMessage("Ready")
```

**Concept quan trọng — Layout hierarchy**:
```
QMainWindow
  └─ centralWidget = QTabWidget
       ├─ Tab 0: CollectTab (QWidget) ← sẽ làm ở Phase 2
       └─ Tab 1: ProcessTab (QWidget) ← sẽ làm ở Phase 4
```

#### 1.6 `recorder/core/__init__.py`
File trống.

### Verification
Chạy `python recorder/main.py` → cửa sổ hiện lên với 2 tab trống, có thể click qua lại giữa tab.

### Cấu trúc thư mục sau Phase 1
```
recorder/
├── __init__.py
├── main.py
├── config.py
├── ui/
│   ├── __init__.py
│   └── main_window.py
└── core/
    └── __init__.py
```

---

## Phase 2: Tab Collect — Hiển thị skeleton realtime

### Mục tiêu
Chạy MockLeapMotionCapture → nhận frame → vẽ skeleton bàn tay live 30fps trên GUI.

### Khái niệm mới cần hiểu

**QThread vs threading.Thread** — Qt có hệ thống thread riêng. Quy tắc vàng: **CHỈ update UI từ main thread**. Nếu background thread muốn update UI → phải dùng Signal.

**QTimer** — Hẹn giờ lặp lại chạy trên main thread. Dùng cho render loop:
```python
self._timer = QTimer(self)
self._timer.setInterval(33)  # 1000ms / 30fps ≈ 33ms
self._timer.timeout.connect(self.render_frame)  # gọi render_frame mỗi 33ms
self._timer.start()
```

**Callback pattern** — `MockLeapMotionCapture` gọi callback trên BACKGROUND thread. Callback này emit signal → signal tự động marshal sang main thread → slot update UI.

**QPixmap/QImage** — Qt hiển thị ảnh qua QLabel.setPixmap(). Pipeline:
```
numpy BGR array → QImage(data, w, h, bytes_per_line, Format_RGB888) → QPixmap → QLabel
```

### Flow dữ liệu chi tiết

```
MockLeapMotionCapture (background thread, 60fps)
  │  callback(timestamp_us, frame_np[42,7])
  ▼
CollectTab._on_lmc_frame()    ← BACKGROUND THREAD, không được update UI ở đây!
  │  self._latest_frame = frame_np    # lưu frame mới nhất (atomic reference)
  │  self.frame_received.emit()        # phát signal (thread-safe)
  ▼
QTimer (main thread, 30fps)
  │  timeout signal → _render_tick()
  ▼
_render_tick():
  │  frame = self._latest_frame        # đọc frame mới nhất
  │  canvas = self._visualizer.render(frame)   # vẽ skeleton → BGR numpy
  │  Chuyển canvas → QPixmap → self._vis_label.setPixmap(...)
  ▼
QLabel hiển thị skeleton trên GUI
```

**Tại sao dùng QTimer 30fps thay vì render mỗi khi có frame?**
MockLMC gửi 60fps nhưng screen chỉ cần 30fps. QTimer gom nhiều frame → chỉ render frame mới nhất → mượt mà, không lag UI.

### File tạo mới

#### 2.1 `recorder/ui/collect_tab.py`

```
Class: CollectTab(QWidget)

  Signals:
    frame_received = pyqtSignal()    # emitted khi background thread có frame mới

  __init__(self, parent=None):
    1. Layout chính: QVBoxLayout
    2. Tạo HandVisualizer (import từ whisper_app.app.ui.visualization)
       self._visualizer = HandVisualizer(width=VIS_WIDTH, height=VIS_HEIGHT)
    3. Tạo QLabel cho skeleton display:
       self._vis_label = QLabel()
       self._vis_label.setFixedSize(VIS_WIDTH, VIS_HEIGHT)
       self._vis_label.setStyleSheet("background-color: #1e1e1e;")
    4. Thêm vào layout

    5. Tạo capture (mock mode):
       from whisper_app.app.hardware.lmc_capture import create_capture
       self._capture = create_capture(
           on_frame_callback=self._on_lmc_frame,
           mock=True,
           fps=MOCK_FPS,
       )

    6. State:
       self._latest_frame = None        # numpy (42, 7) or None
       self._is_capturing = False

    7. Kết nối signal:
       self.frame_received.connect(self._on_frame_signal)   # signal → slot

    8. Tạo render timer:
       self._render_timer = QTimer(self)
       self._render_timer.setInterval(int(1000 / VIS_FPS))  # 33ms
       self._render_timer.timeout.connect(self._render_tick)

  def start_capture(self):
    """Bắt đầu nhận frame từ LMC."""
    self._capture.start()
    self._render_timer.start()
    self._is_capturing = True

  def stop_capture(self):
    """Dừng nhận frame."""
    self._render_timer.stop()
    self._capture.stop()
    self._is_capturing = False

  def _on_lmc_frame(self, timestamp_us: int, frame_np: np.ndarray):
    """CALLBACK — chạy trên BACKGROUND thread. KHÔNG update UI ở đây!"""
    self._latest_frame = frame_np    # lưu reference (thread-safe vì Python GIL)
    self.frame_received.emit()        # phát signal → Qt marshal sang main thread

  def _on_frame_signal(self):
    """SLOT — chạy trên main thread. Không cần làm gì ở đây vì render_tick sẽ đọc."""
    pass    # QTimer sẽ tự render ở tần suất riêng

  def _render_tick(self):
    """SLOT — chạy 30fps trên main thread. Vẽ skeleton."""
    frame = self._latest_frame
    if frame is None:
        return

    # 1. Vẽ skeleton
    canvas = self._visualizer.render(frame)  # BGR numpy (H, W, 3)

    # 2. Chuyển BGR → RGB (Qt dùng RGB)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # 3. Numpy → QImage → QPixmap → QLabel
    h, w, ch = canvas_rgb.shape
    bytes_per_line = ch * w
    q_image = QImage(canvas_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    self._vis_label.setPixmap(QPixmap.fromImage(q_image))
```

**Import cần thiết**:
```python
import cv2
import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from whisper_app.app.ui.visualization import HandVisualizer
from whisper_app.app.hardware.lmc_capture import create_capture
from recorder.config import VIS_WIDTH, VIS_HEIGHT, VIS_FPS, MOCK_FPS
```

### Cập nhật `recorder/ui/main_window.py`

Thay placeholder tab 0 bằng `CollectTab`:
```python
from recorder.ui.collect_tab import CollectTab

# Trong __init__:
self._collect_tab = CollectTab()
tab_widget.addTab(self._collect_tab, "📹 Collect")

# Bắt đầu capture ngay khi mở:
self._collect_tab.start_capture()
```

Và trong `closeEvent(self, event)`:
```python
self._collect_tab.stop_capture()
event.accept()
```

### Khái niệm Numpy cần nắm

**Frame shape `(42, 7)`** nghĩa là:
- 42 hàng = 42 joints (21 tay trái + 21 tay phải)
- 7 cột = [x, y, z, vx, vy, vz, confidence]

**Truy cập**:
- `frame[:21]` → 21 joints tay trái, shape (21, 7)
- `frame[21:]` → 21 joints tay phải, shape (21, 7)
- `frame[0, :3]` → [x, y, z] của wrist trái
- `frame[:, 6]` → confidence của tất cả 42 joints

**dtype=float32** — luôn dùng float32 cho ML data. Mock tạo random: `np.random.randn(42, 7).astype(np.float32)`

### Verification
Chạy `python recorder/main.py` → Tab Collect hiển thị skeleton ngẫu nhiên di chuyển mượt mà ~30fps.

---

## Phase 3: Tab Collect — Ghi gesture

### Mục tiêu
Thêm controls để ghi 1 gesture, đếm ngược 3-2-1, lưu thành .npy file.

### Khái niệm mới

**State Machine** — Quản lý trạng thái app bằng enum:
```
IDLE → (nhấn Record) → COUNTDOWN → (3-2-1 xong) → RECORDING → (nhấn Stop) → SAVING → IDLE
                                                  → (nhấn Cancel) → IDLE
```

Tại sao cần state machine? Vì mỗi trạng thái có behavior khác nhau:
- IDLE: nút Record enabled, nút Stop disabled
- COUNTDOWN: hiện overlay "3... 2... 1..."
- RECORDING: buffer frames vào list, hiện "REC ●" indicator
- SAVING: lưu file, nút disabled

**numpy .npy format** — Format binary của numpy, lưu/đọc cực nhanh:
```python
np.save("path/to/file.npy", array)     # Lưu
array = np.load("path/to/file.npy")    # Đọc
# File .npy giữ nguyên shape, dtype, không mất precision
```

**np.stack** — Ghép list of arrays thành 1 array mới:
```python
frames = [frame1, frame2, frame3]  # mỗi cái shape (42, 7)
sequence = np.stack(frames)         # shape (3, 42, 7) — T=3
```

### File tạo mới

#### 3.1 `recorder/core/session_manager.py`

```
Class: SessionManager

  __init__(self, raw_dir: Path = RAW_DIR, sessions_file: Path = SESSIONS_FILE):
    self._raw_dir = raw_dir
    self._sessions_file = sessions_file
    self._sessions = self._load_sessions()

  def _load_sessions(self) -> dict:
    """Đọc sessions.json, tạo mới nếu chưa có."""
    if self._sessions_file.exists():
        return json.load(open(self._sessions_file))
    return {"recordings": []}

  def _save_sessions(self):
    """Ghi sessions.json."""
    self._sessions_file.parent.mkdir(parents=True, exist_ok=True)
    with open(self._sessions_file, "w") as f:
        json.dump(self._sessions, f, indent=2, ensure_ascii=False)

  def save_recording(self, gloss: str, frames: list[np.ndarray]) -> Path:
    """
    Lưu 1 recording.

    Args:
        gloss: Tên gesture (ví dụ "xin_chao")
        frames: List of numpy arrays, mỗi cái shape (42, 7)

    Returns:
        Path tới file .npy đã lưu

    Logic:
    1. sequence = np.stack(frames)  → shape (T, 42, 7)
    2. Tạo thư mục: raw_dir / gloss / (mkdir parents=True)
    3. Filename: timestamp format "20260311_142301.npy"
       from datetime import datetime
       filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".npy"
    4. np.save(filepath, sequence)
    5. Thêm entry vào sessions dict:
       {
         "gloss": gloss,
         "file": str(filepath.relative_to(self._raw_dir)),
         "num_frames": len(frames),
         "shape": list(sequence.shape),
         "timestamp": datetime.now().isoformat(),
       }
    6. self._save_sessions()
    7. Return filepath
    """

  def get_gloss_counts(self) -> dict:
    """Đếm số recordings cho mỗi gloss. Return: {"xin_chao": 5, "cam_on": 3}"""

  def get_all_glosses(self) -> list:
    """Return danh sách tên gloss đã ghi (sorted)."""
```

**sessions.json format**:
```json
{
  "recordings": [
    {
      "gloss": "xin_chao",
      "file": "xin_chao/20260311_142301.npy",
      "num_frames": 180,
      "shape": [180, 42, 7],
      "timestamp": "2026-03-11T14:23:01"
    }
  ]
}
```

#### 3.2 Cập nhật `recorder/ui/collect_tab.py`

**Thêm UI controls** (thêm vào `__init__`):

```
Layout mới (2 phần: visualization bên trái, controls bên phải):

QHBoxLayout (main)
├── QVBoxLayout (left — 60%)
│   └── _vis_label (QLabel, skeleton display)
└── QVBoxLayout (right — 40%)
    ├── QLabel("Gloss Name:")
    ├── QComboBox (editable=True)    ← nhập mới HOẶC chọn từ lịch sử
    ├── QHBoxLayout
    │   ├── QPushButton("🔴 Record")
    │   ├── QPushButton("⏹ Stop & Save")    ← disabled ban đầu
    │   └── QPushButton("❌ Cancel")          ← disabled ban đầu
    ├── QLabel(_countdown_label, font 72pt, centered)  ← hiện "3", "2", "1", "REC ●"
    ├── QGroupBox("Statistics")
    │   └── QLabel(_stats_label)     ← "xin_chao: 5 samples\ncam_on: 3 samples"
    └── stretch
```

**State machine** (thêm vào class):

```python
from enum import Enum, auto

class RecordState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    SAVING = auto()

# Trong __init__:
self._state = RecordState.IDLE
self._recording_buffer = []      # List[np.ndarray], mỗi cái (42, 7)
self._countdown_value = 0
self._countdown_timer = QTimer(self)
self._countdown_timer.setInterval(1000)  # 1 giây
self._countdown_timer.timeout.connect(self._countdown_tick)
self._session_manager = SessionManager()
```

**Button handlers**:

```
def _on_record_clicked(self):
    if self._state != RecordState.IDLE:
        return
    gloss = self._gloss_combo.currentText().strip()
    if not gloss:
        # Hiện warning (QMessageBox hoặc đổi border đỏ)
        return
    self._state = RecordState.COUNTDOWN
    self._countdown_value = COUNTDOWN_SECONDS  # 3
    self._update_ui_for_state()
    self._countdown_label.setText(str(self._countdown_value))
    self._countdown_timer.start()

def _countdown_tick(self):
    self._countdown_value -= 1
    if self._countdown_value > 0:
        self._countdown_label.setText(str(self._countdown_value))
    else:
        # Countdown xong → bắt đầu ghi
        self._countdown_timer.stop()
        self._countdown_label.setText("● REC")
        self._countdown_label.setStyleSheet("color: red; font-size: 48pt;")
        self._recording_buffer.clear()
        self._state = RecordState.RECORDING
        self._update_ui_for_state()

def _on_stop_clicked(self):
    if self._state != RecordState.RECORDING:
        return
    self._state = RecordState.SAVING
    self._update_ui_for_state()

    gloss = self._gloss_combo.currentText().strip()
    filepath = self._session_manager.save_recording(gloss, self._recording_buffer)
    # Cập nhật stats
    self._update_stats_display()
    self._recording_buffer.clear()
    self._countdown_label.setText("")
    self._state = RecordState.IDLE
    self._update_ui_for_state()

def _on_cancel_clicked(self):
    if self._state in (RecordState.COUNTDOWN, RecordState.RECORDING):
        self._countdown_timer.stop()
        self._recording_buffer.clear()
        self._countdown_label.setText("")
        self._state = RecordState.IDLE
        self._update_ui_for_state()
```

**Sửa `_on_lmc_frame` để buffer khi đang ghi**:

```python
def _on_lmc_frame(self, timestamp_us, frame_np):
    self._latest_frame = frame_np
    # Buffer frame nếu đang recording
    if self._state == RecordState.RECORDING:
        self._recording_buffer.append(frame_np.copy())  # .copy() quan trọng!
    self.frame_received.emit()
```

**Tại sao cần `.copy()`?** Vì MockLeapMotionCapture có thể reuse cùng numpy array cho frame tiếp theo. Nếu không copy, tất cả frames trong buffer sẽ trỏ tới cùng 1 vùng nhớ → dữ liệu sai.

**`_update_ui_for_state()`** — Enable/disable buttons theo state:

```python
def _update_ui_for_state(self):
    s = self._state
    self._record_btn.setEnabled(s == RecordState.IDLE)
    self._stop_btn.setEnabled(s == RecordState.RECORDING)
    self._cancel_btn.setEnabled(s in (RecordState.COUNTDOWN, RecordState.RECORDING))
    self._gloss_combo.setEnabled(s == RecordState.IDLE)
```

### Verification
1. Nhập "xin_chao" → nhấn Record → thấy countdown 3-2-1 → thấy "REC ●" → nhấn Stop
2. Kiểm tra: `recorder/data/raw/xin_chao/` có file .npy
3. Verify shape:
```python
import numpy as np
data = np.load("recorder/data/raw/xin_chao/20260311_142301.npy")
print(data.shape)   # (T, 42, 7) — T = số frames đã ghi
print(data.dtype)    # float32
```
4. Ghi 3 lần → 3 files, sessions.json cập nhật đúng

---

## Phase 4: Tab Process — Scan + Preview

### Mục tiêu
Tab Process quét thư mục raw data, hiển thị thống kê (bao nhiêu samples mỗi gloss, avg frames...).

### Khái niệm mới

**pathlib.Path.glob / rglob** — Tìm files theo pattern:
```python
from pathlib import Path
raw_dir = Path("recorder/data/raw")
for npy_file in raw_dir.rglob("*.npy"):    # tìm tất cả .npy trong mọi subfolder
    print(npy_file.parent.name, npy_file.name)
    # → "xin_chao", "20260311_142301.npy"
```

**QTableWidget** — Bảng dữ liệu trong Qt:
```python
table = QTableWidget()
table.setColumnCount(4)
table.setHorizontalHeaderLabels(["Gloss", "Count", "Avg Frames", "Est. Split"])
table.setRowCount(len(glosses))
table.setItem(row, col, QTableWidgetItem("text"))
```

**QFileDialog** — Dialog chọn thư mục:
```python
dir_path = QFileDialog.getExistingDirectory(self, "Chọn thư mục raw data")
```

### Files tạo mới

#### 4.1 `recorder/core/data_processor.py`

```
Class: DataProcessor

  def scan_raw_directory(self, raw_dir: Path) -> list[dict]:
    """
    Quét thư mục raw, trả về thống kê cho mỗi gloss.

    Logic:
    1. Liệt kê tất cả subfolder trong raw_dir (mỗi subfolder = 1 gloss)
    2. Với mỗi subfolder:
       a. Đếm số file .npy
       b. Load header của mỗi file (np.load với mmap_mode='r' để không load toàn bộ vào RAM):
          data = np.load(filepath, mmap_mode='r')
          shape = data.shape  # (T, 42, 7)
       c. Kiểm tra shape hợp lệ: len(shape) == 3 and shape[1] == 42 and shape[2] == 7
       d. Tính avg_frames = mean([f.shape[0] for f in valid_files])
       e. Tính duration ước tính: avg_frames / 60  # giây, giả sử 60fps

    Returns: [
      {
        "gloss": "xin_chao",
        "count": 5,
        "valid": 5,
        "invalid": 0,
        "avg_frames": 180,
        "avg_duration_s": 3.0,
        "min_frames": 120,
        "max_frames": 240,
      },
      ...
    ]
    """

  def estimate_split(self, stats: list[dict]) -> dict:
    """
    Ước tính số samples cho train/val/test split 80/10/10.

    Returns: {
      "total": 50,
      "train": 40,
      "val": 5,
      "test": 5,
      "glosses": 10,
    }
    """
```

**Tip numpy — mmap_mode='r'**: Khi chỉ cần đọc shape/metadata, dùng memory-mapped mode. Numpy sẽ không load toàn bộ file vào RAM, chỉ đọc header:
```python
data = np.load("file.npy", mmap_mode='r')
print(data.shape)  # Nhanh, không tốn RAM
# data[0] sẽ đọc on-demand từ disk
```

#### 4.2 `recorder/ui/process_tab.py`

```
Class: ProcessTab(QWidget)

  __init__(self):
    Layout:
    QVBoxLayout
    ├── QHBoxLayout (top bar)
    │   ├── QLabel("Input Directory:")
    │   ├── QLineEdit(_dir_input, readonly, text=str(RAW_DIR))
    │   ├── QPushButton("Browse...")  → connect to _on_browse
    │   └── QPushButton("🔍 Scan")   → connect to _on_scan
    ├── QTableWidget(_stats_table)     ← chiếm phần lớn diện tích
    │   columns: Gloss | Count | Valid | Avg Frames | Avg Duration | Min | Max
    ├── QGroupBox("Summary")
    │   └── QLabel(_summary_label)     ← "Total: 50 samples, 10 glosses. Est split: 40/5/5"
    └── QPushButton("▶ Process & Export")  → connect to _on_process (Phase 5)
        (disabled cho tới khi scan xong)

  def _on_browse(self):
    dir_path = QFileDialog.getExistingDirectory(self, "Select raw data directory")
    if dir_path:
        self._dir_input.setText(dir_path)

  def _on_scan(self):
    raw_dir = Path(self._dir_input.text())
    if not raw_dir.exists():
        QMessageBox.warning(self, "Error", f"Directory not found: {raw_dir}")
        return

    processor = DataProcessor()
    stats = processor.scan_raw_directory(raw_dir)

    # Populate table
    self._stats_table.setRowCount(len(stats))
    for row, s in enumerate(stats):
        self._stats_table.setItem(row, 0, QTableWidgetItem(s["gloss"]))
        self._stats_table.setItem(row, 1, QTableWidgetItem(str(s["count"])))
        # ... etc

    # Cập nhật summary
    est = processor.estimate_split(stats)
    self._summary_label.setText(
        f"Total: {est['total']} samples, {est['glosses']} glosses. "
        f"Est split: {est['train']}/{est['val']}/{est['test']}"
    )
    self._process_btn.setEnabled(True)
```

### Cập nhật `recorder/ui/main_window.py`
Thay placeholder tab 1 bằng `ProcessTab`:
```python
from recorder.ui.process_tab import ProcessTab
self._process_tab = ProcessTab()
tab_widget.addTab(self._process_tab, "⚙️ Process")
```

### Verification
1. Sau khi ghi vài samples ở Phase 3 → chuyển sang tab Process
2. Nhấn Scan → bảng hiển thị đúng gloss, count, avg frames
3. Summary hiện đúng total và estimated split

---

## Phase 5: Tab Process — Pipeline xử lý + Export

### Mục tiêu
Chạy full pipeline: resample → compute velocity → normalize → split → export format cho training.

### Khái niệm mới

**QThread worker pattern** — Chạy heavy work trên background thread, cập nhật progress trên UI:

```python
class ProcessWorker(QThread):
    progress = pyqtSignal(int, str)    # (percent, message)
    finished = pyqtSignal(dict)         # kết quả
    error = pyqtSignal(str)             # lỗi

    def __init__(self, raw_dir, output_dir, ...):
        super().__init__()
        self._raw_dir = raw_dir
        self._output_dir = output_dir

    def run(self):
        """CHẠY TRÊN BACKGROUND THREAD. Không được access UI ở đây!"""
        try:
            # ... heavy processing ...
            self.progress.emit(50, "Resampling...")
            # ... more work ...
            self.finished.emit({"train": 40, "val": 5, "test": 5})
        except Exception as e:
            self.error.emit(str(e))
```

Kết nối trong UI:
```python
worker = ProcessWorker(raw_dir, output_dir)
worker.progress.connect(self._on_progress)      # update progress bar
worker.finished.connect(self._on_finished)       # show results
worker.error.connect(self._on_error)             # show error
worker.start()
```

**QProgressBar**:
```python
self._progress = QProgressBar()
self._progress.setRange(0, 100)
self._progress.setValue(50)
```

**Stratified split** — Split data sao cho mỗi gloss có tỉ lệ train/val/test đều:
```python
from sklearn.model_selection import train_test_split
# Hoặc tự implement:
# Với mỗi gloss, shuffle files, lấy 80% đầu cho train, 10% val, 10% test
```

### Thêm vào `recorder/core/data_processor.py`

```
Thêm method vào class DataProcessor:

  def process_and_export(
      self,
      raw_dir: Path,
      output_dir: Path,
      target_fps: int = 60,
      test_ratio: float = 0.1,
      val_ratio: float = 0.1,
      progress_callback: Callable[[int, str], None] = None,
  ) -> dict:
    """
    Full pipeline: raw .npy → processed train/val/test splits.

    Pipeline cho TỪNG file .npy:
    1. Load: data = np.load(filepath)  → shape (T_raw, 42, 7)

    2. Recompute velocities (vì mock data có velocity random):
       Reuse logic từ Whisper_modification/src/data/preprocessing.py
       hoặc tự viết:
         positions = data[:, :, :3]                    # (T, 42, 3)
         dt = 1.0 / target_fps
         velocities = np.gradient(positions, dt, axis=0)  # (T, 42, 3)
         data[:, :, 3:6] = velocities

    3. (KHÔNG normalize ở đây — SpatialNorm + ScaleNorm sẽ áp dụng
       trong DataLoader lúc train, xem SignLanguageDataset.__getitem__)

    Split:
    4. Tạo label_map: {"<blank>": 0, "xin_chao": 1, "cam_on": 2, ...}
       - "<blank>" luôn = 0 (CTC blank token)
       - Các gloss khác đánh số từ 1

    5. Stratified split:
       Với mỗi gloss:
         files = sorted(gloss_dir.glob("*.npy"))
         random.shuffle(files)   # seed=42 cho reproducible
         n = len(files)
         n_test = max(1, round(n * test_ratio))
         n_val = max(1, round(n * val_ratio))
         n_train = n - n_test - n_val
         # Nếu gloss chỉ có 1-2 files → cho tất cả vào train, skip val/test

    Export:
    6. Cho mỗi split (train/val/test):
       output_dir / split / features / sample_00000.npy  → shape (T, 42, 7) float32
       output_dir / split / labels / sample_00000.npy    → shape (1,) int64

       Naming: sample_XXXXX với counter liên tục (00000, 00001, ...)

    7. Lưu label_map.json:
       output_dir / label_map.json

    8. Emit progress: progress_callback(percent, message)
       Chia đều: scan=10%, process=70%, export=20%

    Returns: {
      "train_count": 40,
      "val_count": 5,
      "test_count": 5,
      "num_glosses": 10,
      "label_map": {...},
    }
    """
```

**Data shapes qua pipeline**:
```
Raw input:     (T_variable, 42, 7) float32   ← từ recorder, T khác nhau mỗi file
After velocity: (T, 42, 7) float32            ← velocity recomputed
Exported:      features/(T, 42, 7) float32    ← giữ nguyên T, KHÔNG pad
               labels/(1,) int64              ← single gloss ID

Lúc training (trong SignLanguageDataset.__getitem__):
  → SpatialNorm → ScaleNorm → pad/truncate to 1500 → augment
  → Output: (1500, 42, 7) float32
```

### File tạo mới

#### 5.1 `recorder/core/process_worker.py`

```
Class: ProcessWorker(QThread)

  progress = pyqtSignal(int, str)    # (percent, message)
  finished = pyqtSignal(dict)
  error = pyqtSignal(str)

  __init__(self, raw_dir: Path, output_dir: Path, target_fps: int = 60):
    super().__init__()
    self._raw_dir = raw_dir
    self._output_dir = output_dir
    self._target_fps = target_fps

  def run(self):
    try:
        processor = DataProcessor()
        result = processor.process_and_export(
            raw_dir=self._raw_dir,
            output_dir=self._output_dir,
            target_fps=self._target_fps,
            progress_callback=lambda p, m: self.progress.emit(p, m),
        )
        self.finished.emit(result)
    except Exception as e:
        self.error.emit(str(e))
```

#### 5.2 Cập nhật `recorder/ui/process_tab.py`

Thêm vào layout (dưới Process button):
```
├── QProgressBar(_progress_bar, hidden ban đầu)
├── QTextEdit(_log_text, readonly, font Consolas 9pt, max 200 lines)
```

**Handler cho nút Process & Export**:
```python
def _on_process(self):
    raw_dir = Path(self._dir_input.text())
    output_dir = PROCESSED_DIR  # từ config.py

    self._process_btn.setEnabled(False)
    self._progress_bar.show()
    self._progress_bar.setValue(0)
    self._log_text.clear()

    self._worker = ProcessWorker(raw_dir, output_dir)
    self._worker.progress.connect(self._on_progress)
    self._worker.finished.connect(self._on_finished)
    self._worker.error.connect(self._on_error)
    self._worker.start()

def _on_progress(self, percent, message):
    self._progress_bar.setValue(percent)
    self._log_text.append(message)

def _on_finished(self, result):
    self._progress_bar.setValue(100)
    self._log_text.append(f"\n✅ Done! Train: {result['train_count']}, "
                          f"Val: {result['val_count']}, Test: {result['test_count']}")
    self._process_btn.setEnabled(True)

def _on_error(self, error_msg):
    self._log_text.append(f"\n❌ Error: {error_msg}")
    self._process_btn.setEnabled(True)
    QMessageBox.critical(self, "Processing Error", error_msg)
```

### Verification

1. Output dirs tồn tại:
```python
from pathlib import Path
p = Path("recorder/data/processed")
assert (p / "train" / "features").exists()
assert (p / "train" / "labels").exists()
assert (p / "val" / "features").exists()
assert (p / "test" / "features").exists()
assert (p / "label_map.json").exists()
```

2. File shapes đúng:
```python
import numpy as np, json
feat = np.load("recorder/data/processed/train/features/sample_00000.npy")
label = np.load("recorder/data/processed/train/labels/sample_00000.npy")
print(feat.shape)    # (T, 42, 7) — T biến đổi
print(feat.dtype)    # float32
print(label.shape)   # (1,)
print(label.dtype)   # int64
```

3. label_map.json đúng format:
```python
with open("recorder/data/processed/label_map.json") as f:
    lm = json.load(f)
assert lm["<blank>"] == 0
assert all(isinstance(v, int) for v in lm.values())
```

4. Load vào SignLanguageDataset không lỗi:
```python
import sys; sys.path.insert(0, "Whisper_modification")
from src.data.dataset import SignLanguageDataset
ds = SignLanguageDataset(data_dir="recorder/data/processed", split="train")
sample = ds[0]
print(sample['features'].shape)  # (1500, 42, 7) — đã pad
print(sample['labels'].shape)    # (1,)
```

---

## Tổng kết cấu trúc thư mục cuối cùng

```
recorder/
├── __init__.py
├── main.py                      ← entry point
├── config.py                    ← paths, constants
├── core/
│   ├── __init__.py
│   ├── session_manager.py       ← lưu .npy, quản lý sessions.json
│   ├── data_processor.py        ← scan, validate, process pipeline
│   └── process_worker.py        ← QThread wrapper cho processing
├── ui/
│   ├── __init__.py
│   ├── main_window.py           ← QMainWindow + QTabWidget
│   ├── collect_tab.py           ← Tab ghi gesture (Phase 2-3)
│   └── process_tab.py           ← Tab xử lý + export (Phase 4-5)
└── data/                         ← TẠO BỞI APP, KHÔNG COMMIT VÀO GIT
    ├── raw/
    │   ├── xin_chao/
    │   │   ├── 20260311_142301.npy   (T, 42, 7) float32, mm
    │   │   └── 20260311_142315.npy
    │   └── cam_on/
    ├── processed/
    │   ├── train/features/*.npy
    │   ├── train/labels/*.npy
    │   ├── val/...
    │   ├── test/...
    │   └── label_map.json
    └── sessions.json
```

## Quyết định
- App standalone (không tích hợp whisper_app)
- Mock mode mặc định (auto-detect LMC hardware)
- Dữ liệu raw lưu mm (không normalize) → pipeline training xử lý đúng
- SpatialNorm + ScaleNorm KHÔNG áp dụng khi lưu raw — chỉ áp dụng trong SignLanguageDataset.__getitem__ lúc train
- label_map.json: "<blank>" = 0, glosses bắt đầu từ 1 (CTC convention)
- Velocity được recompute từ positions khi export (np.gradient), không dùng palm velocity từ mock

## Relevant files (reuse/tham khảo)
- `whisper_app/main.py` — pattern khởi tạo QApplication
- `whisper_app/app/ui/main_window.py` — layout pattern, signal-slot, QTimer render
- `whisper_app/app/ui/visualization.py` — HandVisualizer.render(frame) nhận (42,7) → BGR canvas
- `whisper_app/app/hardware/lmc_capture.py` — MockLeapMotionCapture, create_capture() factory
- `Whisper_modification/src/data/dataset.py` — SignLanguageDataset expects features/*.npy + labels/*.npy + label_map.json
- `Whisper_modification/src/data/preprocessing.py` — resample_to_fixed_rate(), preprocess_sequence()
- `Whisper_modification/src/data/normalization.py` — SpatialNormalizer, ScaleNormalizer (dùng trong DataLoader, không dùng khi lưu raw)
- `Whisper_modification/scripts/prepare_vsl_data.py` — label_map creation pattern, split logic
