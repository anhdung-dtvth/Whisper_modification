"""
Entry point for WhisperSign Data Recorder.
"""
from pathlib import Path
from PyQt5.QtWidgets import QApplication
import sys

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from recorder.ui.main_window import MainWindow
    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
