import cv2
import numpy as np
from enum import Enum, auto
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QGroupBox, QMessageBox, QCheckBox,
)
from recorder.utils.visualization import HandVisualizer
from recorder.utils.lmc_capture import create_capture
from recorder.config import VIS_WIDTH, VIS_HEIGHT, VIS_FPS, MOCK_FPS, DEFAULT_FPS, COUNTDOWN_SECONDS
from recorder.core.session_manager import SessionManager


class RecordState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    RECORDING = auto()
    SAVING = auto()


class CollectTab(QWidget):
    frame_received = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

        # --- Layout: left (visualization) + right (controls) ---
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left: visualization
        left_layout = QVBoxLayout()
        self._visualizer = HandVisualizer(VIS_WIDTH, VIS_HEIGHT)
        self._vis_label = QLabel(self)
        self._vis_label.setFixedSize(VIS_WIDTH, VIS_HEIGHT)
        self._vis_label.setStyleSheet("background-color: #1e1e1e;")
        left_layout.addWidget(self._vis_label)
        left_layout.addStretch()
        main_layout.addLayout(left_layout, 6)

        # Right: controls
        right_layout = QVBoxLayout()

        # Mock mode toggle
        self._mock_checkbox = QCheckBox("Mock Mode (no hardware)")
        self._mock_checkbox.setChecked(True)
        self._mock_checkbox.toggled.connect(self._on_mock_toggled)
        right_layout.addWidget(self._mock_checkbox)

        self._mode_label = QLabel("Mode: Mock")
        self._mode_label.setStyleSheet("color: orange; font-weight: bold;")
        right_layout.addWidget(self._mode_label)

        right_layout.addWidget(QLabel("Gloss Name:"))
        self._gloss_combo = QComboBox()
        self._gloss_combo.setEditable(True)
        right_layout.addWidget(self._gloss_combo)

        btn_layout = QHBoxLayout()
        self._record_btn = QPushButton("🔴 Record")
        self._stop_btn = QPushButton("⏹ Stop && Save")
        self._cancel_btn = QPushButton("❌ Cancel")
        self._stop_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        btn_layout.addWidget(self._record_btn)
        btn_layout.addWidget(self._stop_btn)
        btn_layout.addWidget(self._cancel_btn)
        right_layout.addLayout(btn_layout)

        self._countdown_label = QLabel("")
        self._countdown_label.setAlignment(Qt.AlignCenter)
        self._countdown_label.setFont(QFont("Arial", 72))
        self._countdown_label.setMinimumHeight(120)
        right_layout.addWidget(self._countdown_label)

        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        self._stats_label = QLabel("No recordings yet.")
        stats_layout.addWidget(self._stats_label)
        stats_group.setLayout(stats_layout)
        right_layout.addWidget(stats_group)

        right_layout.addStretch()
        main_layout.addLayout(right_layout, 4)

        # --- Capture ---
        self._mock = True
        self._capture = create_capture(
            on_frame_callback=self._on_lmc_frame,
            mock=self._mock,
            fps=MOCK_FPS,
        )

        # --- State ---
        self._latest_frame = None
        self._is_capturing = False
        self._state = RecordState.IDLE
        self._recording_buffer = []
        self._countdown_value = 0

        # --- Session manager ---
        self._session_manager = SessionManager()

        # --- Signals / Timers ---
        self.frame_received.connect(self._on_frame_signal)

        self._render_timer = QTimer(self)
        self._render_timer.setInterval(int(1000 / VIS_FPS))
        self._render_timer.timeout.connect(self._render_tick)

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._countdown_tick)

        # --- Button connections ---
        self._record_btn.clicked.connect(self._on_record_clicked)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)

        # --- Init UI ---
        self._update_stats_display()
        self._refresh_gloss_combo()

    # ---- Capture control ----

    def start_capture(self):
        self._capture.start()
        self._render_timer.start()
        self._is_capturing = True

    def stop_capture(self):
        self._render_timer.stop()
        self._capture.stop()
        self._is_capturing = False

    def _on_mock_toggled(self, checked: bool):
        """Switch between mock and real LMC capture."""
        if self._state != RecordState.IDLE:
            self._mock_checkbox.setChecked(self._mock)
            return

        was_capturing = self._is_capturing
        if was_capturing:
            self.stop_capture()

        self._mock = checked
        self._latest_frame = None
        self._capture = create_capture(
            on_frame_callback=self._on_lmc_frame,
            mock=self._mock,
            fps=MOCK_FPS if self._mock else DEFAULT_FPS,
        )

        if self._mock:
            self._mode_label.setText("Mode: Mock")
            self._mode_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self._mode_label.setText("Mode: Leap Motion")
            self._mode_label.setStyleSheet("color: #00cc00; font-weight: bold;")

        if was_capturing:
            self.start_capture()

    # ---- Frame handling ----

    def _on_lmc_frame(self, timestamp_us, frame_np):
        """CALLBACK — runs on BACKGROUND thread."""
        self._latest_frame = frame_np
        if self._state == RecordState.RECORDING:
            self._recording_buffer.append(frame_np.copy())
        self.frame_received.emit(frame_np)

    @pyqtSlot(object)
    def _on_frame_signal(self, frame_np):
        pass

    def _render_tick(self):
        frame = self._latest_frame
        if frame is None:
            return

        canvas = self._visualizer.render(frame)
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = canvas_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(canvas_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self._vis_label.setPixmap(QPixmap.fromImage(q_image))

    # ---- Recording state machine ----

    def _on_record_clicked(self):
        if self._state != RecordState.IDLE:
            return
        gloss = self._gloss_combo.currentText().strip()
        if not gloss:
            QMessageBox.warning(self, "Warning", "Please enter a gloss name.")
            return
        self._state = RecordState.COUNTDOWN
        self._countdown_value = COUNTDOWN_SECONDS
        self._countdown_label.setStyleSheet("color: white; font-size: 72pt;")
        self._countdown_label.setText(str(self._countdown_value))
        self._update_ui_for_state()
        self._countdown_timer.start()

    def _countdown_tick(self):
        self._countdown_value -= 1
        if self._countdown_value > 0:
            self._countdown_label.setText(str(self._countdown_value))
        else:
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
        if self._recording_buffer:
            self._session_manager.save_recording(gloss, self._recording_buffer)

        self._recording_buffer.clear()
        self._countdown_label.setText("")
        self._update_stats_display()
        self._refresh_gloss_combo()
        self._state = RecordState.IDLE
        self._update_ui_for_state()

    def _on_cancel_clicked(self):
        if self._state in (RecordState.COUNTDOWN, RecordState.RECORDING):
            self._countdown_timer.stop()
            self._recording_buffer.clear()
            self._countdown_label.setText("")
            self._state = RecordState.IDLE
            self._update_ui_for_state()

    def _update_ui_for_state(self):
        s = self._state
        self._record_btn.setEnabled(s == RecordState.IDLE)
        self._stop_btn.setEnabled(s == RecordState.RECORDING)
        self._cancel_btn.setEnabled(s in (RecordState.COUNTDOWN, RecordState.RECORDING))
        self._gloss_combo.setEnabled(s == RecordState.IDLE)

    # ---- Stats / Gloss helpers ----

    def _update_stats_display(self):
        counts = self._session_manager.get_gloss_counts()
        if not counts:
            self._stats_label.setText("No recordings yet.")
            return
        lines = [f"{g}: {c} samples" for g, c in sorted(counts.items())]
        self._stats_label.setText("\n".join(lines))

    def _refresh_gloss_combo(self):
        current_text = self._gloss_combo.currentText()
        self._gloss_combo.clear()
        glosses = self._session_manager.get_all_glosses()
        self._gloss_combo.addItems(glosses)
        if current_text:
            self._gloss_combo.setEditText(current_text)