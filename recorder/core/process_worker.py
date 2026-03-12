from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal
from recorder.core.data_processor import DataProcessor


class ProcessWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, raw_dir: Path, output_dir: Path, target_fps: int = 60):
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
