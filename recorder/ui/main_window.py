from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget,
)
from recorder.ui.collect_tab import CollectTab
from recorder.ui.process_tab import ProcessTab


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("WhisperSign Data Recorder")
        self.resize(1000, 700)

        tab_widget = QTabWidget()
        self.setCentralWidget(tab_widget)
        self.statusBar().showMessage("Ready")

        self._collect_tab = CollectTab()
        tab_widget.addTab(self._collect_tab, "📹 Collect")

        self._process_tab = ProcessTab()
        tab_widget.addTab(self._process_tab, "⚙️ Process")

        self._collect_tab.start_capture()

    def closeEvent(self, event):
        self._collect_tab.stop_capture()
        event.accept()