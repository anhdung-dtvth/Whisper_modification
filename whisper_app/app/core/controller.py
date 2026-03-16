"""
Application Controller — orchestrates all components.

Data flow:
  LeapMotionCapture (callback) → FrameBuffer → RealTimePreprocessor →
  InferenceService (periodic) → MainWindow (Qt signal)

Additionally, each raw frame is sent to the UI for skeleton visualization.
"""

import logging
import numpy as np
from typing import Dict, Any

from ..hardware.lmc_capture import create_capture
from ..hardware.preprocessing import FrameBuffer, RealTimePreprocessor
from ..services.inference_service import InferenceService
from ..ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class AppController:
    """
    Central orchestrator wiring capture, preprocessing, inference, and UI.
    """

    def __init__(self, config: Dict[str, Any], window: MainWindow):
        self._config = config
        self._window = window

        hw = config["hardware"]
        mdl = config["model"]
        pre = config["preprocessing"]

        fps = hw.get("target_fps", 60)
        buffer_duration = hw.get("buffer_duration", 2.0)
        max_frames = int(buffer_duration * fps)

        # Frame buffer
        self._buffer = FrameBuffer(max_frames=max_frames, fps=fps)

        # Preprocessor
        self._preprocessor = RealTimePreprocessor(
            smoothing_window=pre.get("smoothing_window", 5),
            spatial_normalization=pre.get("spatial_normalization", True),
            scale_normalization=pre.get("scale_normalization", True),
            fps=fps,
        )

        # Inference service
        self._inference = InferenceService(
            checkpoint_path=mdl.get("checkpoint_path", ""),
            vocab_path=mdl.get("vocab_path"),
            device=mdl.get("device", "cuda"),
            window_duration=mdl.get("window_duration", 2.0),
            sample_rate=fps,
            inference_interval=mdl.get("inference_interval", 0.5),
        )

        # Capture (created on start, destroyed on stop)
        self._capture = None
        self._mock = not self._is_leap_available()

        # Wire UI callbacks
        self._window.set_callbacks(
            on_start=self.start,
            on_stop=self.stop,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self):
        """Load model and update UI status (call before showing window)."""
        self._inference.load_model()
        self._window.update_status(
            model_loaded=self._inference.is_loaded,
            device=self._inference.device,
            lmc_connected=False,
        )

    def start(self):
        """Start capture → preprocessing → inference pipeline."""
        logger.info("Starting pipeline...")

        # Create capture
        self._capture = create_capture(
            on_frame_callback=self._on_frame,
            mock=self._mock,
            fps=self._config["hardware"].get("target_fps", 60),
        )
        self._capture.start()

        # Start periodic inference
        self._inference.start_periodic_inference(
            get_window_fn=self._get_processed_window,
            on_result_fn=self._on_inference_result,
        )

        self._window.set_running(True)
        self._window.update_status(lmc_connected=True)
        logger.info("Pipeline running.")

    def stop(self):
        """Stop all components."""
        logger.info("Stopping pipeline...")

        self._inference.stop_periodic_inference()

        if self._capture is not None:
            self._capture.stop()
            self._capture = None

        self._buffer.clear()
        self._window.set_running(False)
        self._window.update_status(lmc_connected=False)
        logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_frame(self, timestamp_us: int, frame: np.ndarray):
        """
        Called from LMC capture thread for every new frame.
        Pushes to buffer and forwards to UI for visualization.
        """
        self._buffer.add_frame(timestamp_us, frame)
        # Send frame to UI (thread-safe via Qt signal)
        self._window.on_frame(frame)

    def _get_processed_window(self):
        """
        Called by inference thread to get preprocessed data.
        Returns (features, length).
        """
        data, length = self._buffer.get_window()
        if length > 0:
            data, length = self._preprocessor.process(data, length)
        return data, length

    def _on_inference_result(self, token_ids, text, confidence):
        """Called by inference thread when a prediction is ready."""
        self._window.on_prediction(text, confidence)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_leap_available() -> bool:
        try:
            import leap
            return True
        except ImportError:
            return False
