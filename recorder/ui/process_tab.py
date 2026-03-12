from pathlib import Path
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem,
    QGroupBox, QFileDialog, QMessageBox, QProgressBar, QTextEdit,
    QHeaderView,
)
from recorder.config import RAW_DIR, PROCESSED_DIR
from recorder.core.data_processor import DataProcessor
from recorder.core.process_worker import ProcessWorker


class ProcessTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- Top bar: directory selection ---
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Input Directory:"))
        self._dir_input = QLineEdit(str(RAW_DIR))
        self._dir_input.setReadOnly(True)
        top_layout.addWidget(self._dir_input, 1)
        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.clicked.connect(self._on_browse)
        top_layout.addWidget(self._browse_btn)
        self._scan_btn = QPushButton("🔍 Scan")
        self._scan_btn.clicked.connect(self._on_scan)
        top_layout.addWidget(self._scan_btn)
        layout.addLayout(top_layout)

        # --- Stats table ---
        self._stats_table = QTableWidget()
        self._stats_table.setColumnCount(7)
        self._stats_table.setHorizontalHeaderLabels(
            ["Gloss", "Count", "Valid", "Avg Frames", "Avg Duration", "Min", "Max"]
        )
        self._stats_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self._stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._stats_table, 1)

        # --- Summary ---
        summary_group = QGroupBox("Summary")
        summary_layout = QVBoxLayout()
        self._summary_label = QLabel("No data scanned yet.")
        summary_layout.addWidget(self._summary_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        # --- Process button ---
        self._process_btn = QPushButton("▶ Process && Export")
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._on_process)
        layout.addWidget(self._process_btn)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.hide()
        layout.addWidget(self._progress_bar)

        # --- Log text ---
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(150)
        self._log_text.setStyleSheet("font-family: Consolas; font-size: 9pt;")
        layout.addWidget(self._log_text)

        self._worker = None

    # ---- Actions ----

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

        self._stats_table.setRowCount(len(stats))
        for row, s in enumerate(stats):
            self._stats_table.setItem(row, 0, QTableWidgetItem(s["gloss"]))
            self._stats_table.setItem(row, 1, QTableWidgetItem(str(s["count"])))
            self._stats_table.setItem(row, 2, QTableWidgetItem(str(s["valid"])))
            self._stats_table.setItem(row, 3, QTableWidgetItem(str(s["avg_frames"])))
            self._stats_table.setItem(row, 4, QTableWidgetItem(f"{s['avg_duration_s']:.2f}s"))
            self._stats_table.setItem(row, 5, QTableWidgetItem(str(s["min_frames"])))
            self._stats_table.setItem(row, 6, QTableWidgetItem(str(s["max_frames"])))

        est = processor.estimate_split(stats)
        self._summary_label.setText(
            f"Total: {est['total']} samples, {est['glosses']} glosses. "
            f"Est split: {est['train']}/{est['val']}/{est['test']}"
        )
        self._process_btn.setEnabled(est["total"] > 0)

    def _on_process(self):
        raw_dir = Path(self._dir_input.text())
        output_dir = PROCESSED_DIR

        self._process_btn.setEnabled(False)
        self._scan_btn.setEnabled(False)
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
        self._log_text.append(
            f"\n✅ Done! Train: {result['train_count']}, "
            f"Val: {result['val_count']}, Test: {result['test_count']}, "
            f"Glosses: {result['num_glosses']}"
        )
        self._process_btn.setEnabled(True)
        self._scan_btn.setEnabled(True)

    def _on_error(self, error_msg):
        self._log_text.append(f"\n❌ Error: {error_msg}")
        self._process_btn.setEnabled(True)
        self._scan_btn.setEnabled(True)
        self._progress_bar.hide()
        QMessageBox.critical(self, "Processing Error", error_msg)
