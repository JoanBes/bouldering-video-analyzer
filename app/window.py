"""Desktop interface for processing, previewing and exporting climbing videos."""

import os
import math
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import cv2
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QMainWindow, QMessageBox, QPushButton, QProgressBar, QScrollArea, QSizePolicy,
    QSlider, QVBoxLayout, QWidget,
)

from climbing_analyzer import (
    HOLD_COLORS, STATUS_COLORS, STATUS_NAMES, ProcessingCancelled, ROOT, analyze_video, map_wall,
)


STYLE = """
QWidget { background: #101721; color: #e8edf5; font-family: "DejaVu Sans"; font-size: 13px; }
QMainWindow { background: #101721; }
QFrame#sidebar, QFrame#playerCard { background: #182232; border: 1px solid #29394e; border-radius: 14px; }
QFrame#sidebar QLabel, QFrame#playerCard QLabel { background: transparent; }
QLabel#brand { font-size: 21px; font-weight: 700; color: #f4f8ff; }
QLabel#eyebrow { color: #69d7bd; font-weight: 700; font-size: 11px; }
QLabel#heading { font-size: 15px; font-weight: 700; color: #f4f8ff; }
QLabel#muted { color: #a7b5c8; }
QFrame#playerCard QLabel#video { background: #090f17; border: 1px solid #2a394b; border-radius: 8px; color: #8d9bad; }
QPushButton { background: #26364a; border: 1px solid #3a4d64; border-radius: 8px; padding: 10px 14px; font-weight: 600; }
QPushButton:hover { background: #344a62; }
QPushButton:disabled { background: #202b39; color: #657185; border-color: #29384b; }
QPushButton#primary { background: #45c8a9; border: 1px solid #45c8a9; color: #0c2020; }
QPushButton#primary:hover { background: #70dbc1; }
QPushButton#primary:disabled { background: #28443e; color: #7c9b94; border-color: #28443e; }
QComboBox { background: #26364a; border: 1px solid #3a4d64; border-radius: 6px; padding: 5px; }
QProgressBar { background: #0e1621; border: 0; border-radius: 5px; height: 10px; text-align: center; }
QProgressBar::chunk { background: #45c8a9; border-radius: 5px; }
QSlider::groove:horizontal { height: 5px; background: #34445a; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #45c8a9; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; background: #e8f7f3; }
"""


class AnalysisWorker(QThread):
    progress_changed = Signal(str, int, int)
    completed = Signal(str, str, int, object, str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, source: Path, directory: Path, parent=None, pose_size: int = 640):
        super().__init__(parent)
        self.source = source
        self.directory = directory
        self.pose_size = pose_size
        self.stop_requested = threading.Event()

    def cancel(self) -> None:
        self.stop_requested.set()

    def run(self) -> None:
        try:
            for path in (ROOT / "models" / "best_holds.pt", ROOT / "models" / "yolov8m-pose.pt"):
                if not path.is_file():
                    raise FileNotFoundError(f"Model not found: {path}")
            from ultralytics import YOLO

            def progress(stage: str, current: int, total: int) -> None:
                self.progress_changed.emit(stage, current, total)

            self.progress_changed.emit("load", 0, 0)
            holds_model = YOLO(str(ROOT / "models" / "best_holds.pt"))
            map_path = self.directory / "wall_map.json"
            holds = map_wall(self.source, holds_model, map_path, progress=progress,
                             cancelled=self.stop_requested.is_set)
            if self.stop_requested.is_set():
                raise ProcessingCancelled()
            pose_model = YOLO(str(ROOT / "models" / "yolov8m-pose.pt"))
            video_path = self.directory / "analyzed.mp4"
            feedback: list[list[tuple[str, str]]] = []
            summaries: list[str] = []
            frames = analyze_video(self.source, video_path, holds, pose_model, 0.15,
                                   progress=progress, cancelled=self.stop_requested.is_set,
                                   enhanced=True, feedback_callback=feedback.append,
                                   pose_size=self.pose_size, summary_callback=summaries.append)
            if len(summaries) != 1:
                raise RuntimeError("Analysis did not produce an overall recommendation")
            self.completed.emit(str(video_path), str(map_path), frames, feedback, summaries[0])
        except ProcessingCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class VideoPlayer(QWidget):
    """Local video player; decoded frames never leave the application."""

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture = None
        self.original_capture = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 0.0
        self.frame_height = 0
        self.display_height: int | None = None
        self.image = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self.display = QLabel("Select a video to get started")
        self.display.setObjectName("video")
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setMinimumSize(320, 260)
        self.display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.display, 1)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶  Play")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self.toggle)
        controls.addWidget(self.play_button)
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setObjectName("muted")
        controls.addWidget(self.time_label)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setEnabled(False)
        self.seek.sliderReleased.connect(self.seek_to_slider)
        controls.addWidget(self.seek, 1)
        controls.addWidget(QLabel("Speed"))
        self.speed = QComboBox()
        for value in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentIndex(3)
        self.speed.currentIndexChanged.connect(self.update_speed)
        controls.addWidget(self.speed)
        self.overlays = QCheckBox("Show analysis")
        self.overlays.setChecked(True)
        self.overlays.setEnabled(False)
        self.overlays.toggled.connect(self.switch_video)
        controls.addWidget(self.overlays)
        layout.addLayout(controls)

    def load(self, path: Path, display_height: int | None = None,
             original_path: Path | None = None) -> None:
        self.stop()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise OSError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or frames <= 0:
            capture.release()
            raise ValueError("The video does not have a valid duration or frame rate")
        if height <= 0 or (display_height is not None and not 0 < display_height <= height):
            capture.release()
            raise ValueError("The video does not have a valid display height")
        original_capture = None
        if original_path is not None:
            original_capture = cv2.VideoCapture(str(original_path))
            if not original_capture.isOpened():
                original_capture.release()
                capture.release()
                raise OSError(f"Cannot open original video: {original_path}")
            original_fps = float(original_capture.get(cv2.CAP_PROP_FPS))
            original_frames = int(original_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if (not math.isfinite(original_fps) or abs(original_fps - fps) > 0.01
                    or original_frames != frames):
                original_capture.release()
                capture.release()
                raise ValueError("Original and analyzed videos have different durations or frame rates")
        self.capture = capture
        self.original_capture = original_capture
        self.overlays.blockSignals(True)
        self.overlays.setChecked(True)
        self.overlays.blockSignals(False)
        self.overlays.setEnabled(original_capture is not None)
        self.fps = fps
        self.total_frames = frames
        self.frame_height = height
        self.display_height = display_height
        self.seek.setRange(0, frames - 1)
        self.seek.setEnabled(True)
        self.play_button.setEnabled(True)
        if not self.show_frame(0):
            self.stop()
            raise OSError("Could not read the first video frame")

    def stop(self) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        if self.original_capture is not None:
            self.original_capture.release()
            self.original_capture = None
        self.overlays.setEnabled(False)
        self.image = None
        self.play_button.setText("▶  Play")
        self.play_button.setEnabled(False)
        self.seek.setEnabled(False)

    def toggle(self) -> None:
        if self.capture is None:
            return
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("▶  Play")
        else:
            if self.current_frame >= self.total_frames - 1:
                self.show_frame(0)
            self.timer.start(self.playback_interval())
            self.play_button.setText("❚❚  Pause")

    def playback_interval(self) -> int:
        return max(1, round(1000 / (self.fps * self.speed.currentData())))

    def update_speed(self) -> None:
        if self.timer.isActive():
            self.timer.setInterval(self.playback_interval())

    @staticmethod
    def timestamp(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def event_timestamp(seconds: float) -> str:
        milliseconds = max(0, round(seconds * 1000))
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    def show_frame(self, index: int, sequential: bool = False) -> bool:
        if self.capture is None:
            return False
        capture = self.capture if self.overlays.isChecked() or self.original_capture is None else self.original_capture
        if not sequential:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            self.timer.stop()
            self.play_button.setText("▶  Play")
            return False
        self.current_frame = index
        if capture is self.capture and self.display_height is not None:
            frame = frame[:self.display_height, :]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        self.image = QImage(rgb.data, width, height, rgb.strides[0],
                            QImage.Format.Format_RGB888).copy()
        self.refresh_image()
        self.seek.blockSignals(True)
        self.seek.setValue(index)
        self.seek.blockSignals(False)
        self.time_label.setText(
            f"{self.timestamp(index / self.fps)} / {self.timestamp(self.total_frames / self.fps)}"
        )
        self.frame_changed.emit(index)
        return True

    def switch_video(self, checked: bool) -> None:
        if self.capture is not None and not self.show_frame(self.current_frame):
            self.overlays.blockSignals(True)
            self.overlays.setChecked(not checked)
            self.overlays.blockSignals(False)
            self.show_frame(self.current_frame)
            QMessageBox.warning(self, "Playback error", "Could not switch video views.")

    def next_frame(self) -> None:
        if self.current_frame + 1 >= self.total_frames:
            self.timer.stop()
            self.play_button.setText("▶  Play")
        elif not self.show_frame(self.current_frame + 1, sequential=True):
            QMessageBox.warning(self, "Playback error",
                                "Could not read the next video frame.")

    def seek_to_slider(self) -> None:
        if not self.show_frame(self.seek.value()):
            QMessageBox.warning(self, "Playback error", "Could not seek to that position.")

    def refresh_image(self) -> None:
        if self.image is not None:
            self.display.setPixmap(QPixmap.fromImage(self.image).scaled(
                self.display.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refresh_image()


def export_video(source: Path, destination: Path, protected: set[Path]) -> None:
    """Copy to a temporary sibling first, preserving any existing target on failure."""
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError("The destination cannot replace the original video or model files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".climbing-export-", suffix=".mp4",
                                         dir=destination.parent, delete=False) as handle:
            temp_path = Path(handle.name)
        shutil.copyfile(source, temp_path)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crux · Climbing Analysis")
        self.resize(1450, 820)
        self.setMinimumSize(1100, 700)
        self.source: Path | None = None
        self.source_height = 0
        self.result: Path | None = None
        self.feedback: list[list[tuple[str, str]]] = []
        self.overall_recommendation = ""
        self.feedback_cursor = 0
        self.active_feedback: set[tuple[str, str]] = set()
        self.worker: AnalysisWorker | None = None
        self.work_directory: tempfile.TemporaryDirectory | None = None
        self.closing = False

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(26, 22, 26, 22)
        outer.setSpacing(18)
        brand = QLabel("CRUX  /  CLIMBING ANALYZER")
        brand.setObjectName("brand")
        outer.addWidget(brand)
        subtitle = QLabel("Analyze your movement, review your technique, and export the result.")
        subtitle.setObjectName("muted")
        outer.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(18)
        outer.addLayout(content, 1)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(310)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(20, 20, 20, 20)
        side.setSpacing(12)
        eyebrow = QLabel("01  /  YOUR VIDEO")
        eyebrow.setObjectName("eyebrow")
        side.addWidget(eyebrow)
        title = QLabel("Choose a recording")
        title.setObjectName("heading")
        title.setWordWrap(True)
        side.addWidget(title)
        hint = QLabel("Use a fixed camera and show the clear wall at the start to identify holds.")
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        side.addWidget(hint)
        self.select_button = QPushButton("Open video")
        self.select_button.clicked.connect(self.select_video)
        side.addWidget(self.select_button)
        self.file_label = QLabel("No video selected")
        self.file_label.setObjectName("muted")
        self.file_label.setWordWrap(True)
        side.addWidget(self.file_label)
        side.addSpacing(8)
        eyebrow2 = QLabel("02  /  ANALYSIS")
        eyebrow2.setObjectName("eyebrow")
        side.addWidget(eyebrow2)
        quality_label = QLabel("Pose analysis")
        quality_label.setObjectName("muted")
        side.addWidget(quality_label)
        self.pose_quality = QComboBox()
        self.pose_quality.addItem("Fast · 640 px", 640)
        self.pose_quality.addItem("Detailed · 1024 px", 1024)
        self.pose_quality.setToolTip(
            "Fast is quicker; detailed may detect more pose detail but takes longer."
        )
        side.addWidget(self.pose_quality)
        self.process_button = QPushButton("Start analysis")
        self.process_button.setObjectName("primary")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.start_analysis)
        side.addWidget(self.process_button)
        self.cancel_button = QPushButton("Cancel analysis")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        side.addWidget(self.cancel_button)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        side.addWidget(self.progress)
        self.status = QLabel("Ready to start")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        side.addStretch(1)
        legend = QLabel(
            '<span style="color:#39cc94">●</span>  Green · positive cue<br>'
            '<span style="color:#f4b947">●</span>  Amber · recommendation<br>'
            '<span style="color:#f36769">●</span>  Red · needs attention<br>'
            '<span style="color:#8794a6">●</span>  Gray · no assessment'
        )
        legend.setObjectName("muted")
        legend.setWordWrap(True)
        side.addWidget(legend)
        note = QLabel("2D estimates are not a substitute for coaching. Exported video has no audio.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        side.addWidget(note)
        content.addWidget(sidebar)

        card = QFrame()
        card.setObjectName("playerCard")
        right = QVBoxLayout(card)
        right.setContentsMargins(20, 20, 20, 20)
        right.setSpacing(14)
        top = QHBoxLayout()
        self.preview_title = QLabel("Preview")
        self.preview_title.setObjectName("heading")
        top.addWidget(self.preview_title)
        top.addStretch(1)
        self.export_button = QPushButton("Export video")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_result)
        top.addWidget(self.export_button)
        right.addLayout(top)
        hold_title = QLabel("Hold box colors")
        hold_title.setObjectName("muted")
        right.addWidget(hold_title)
        hold_legend = QGridLayout()
        hold_legend.setContentsMargins(0, 0, 0, 0)
        hold_legend.setHorizontalSpacing(12)
        hold_legend.setVerticalSpacing(4)
        for index, (category, color) in enumerate(HOLD_COLORS.items()):
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            label = QLabel(f'<span style="color:{hex_color}">■</span> {category}')
            label.setObjectName("muted")
            hold_legend.addWidget(label, index // 3, index % 3)
        right.addLayout(hold_legend)
        self.player = VideoPlayer()
        right.addWidget(self.player, 1)
        self.player.frame_changed.connect(self.show_feedback)
        content.addWidget(card, 1)

        history_card = QFrame()
        history_card.setObjectName("sidebar")
        history_card.setFixedWidth(280)
        history = QVBoxLayout(history_card)
        history.setContentsMargins(16, 20, 16, 20)
        history.setSpacing(10)
        history_title = QLabel("Feedback history")
        history_title.setObjectName("heading")
        history.addWidget(history_title)
        history_hint = QLabel("Advice appears as you play or seek through the video.")
        history_hint.setObjectName("muted")
        history_hint.setWordWrap(True)
        history.addWidget(history_hint)
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "background: #182232; color: #e8edf5; border: 2px solid #45c8a9; "
            "border-radius: 7px; padding: 8px;"
        )
        self.summary_label.hide()
        history.addWidget(self.summary_label)
        self.feedback_empty = QLabel("No feedback yet.")
        self.feedback_empty.setObjectName("muted")
        self.feedback_empty.setWordWrap(True)
        history.addWidget(self.feedback_empty)
        self.feedback_scroll = QScrollArea()
        self.feedback_scroll.setWidgetResizable(True)
        self.feedback_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.feedback_container = QWidget()
        self.feedback_layout = QVBoxLayout(self.feedback_container)
        self.feedback_layout.setContentsMargins(0, 0, 0, 0)
        self.feedback_layout.setSpacing(8)
        self.feedback_layout.addStretch(1)
        self.feedback_scroll.setWidget(self.feedback_container)
        self.follow_feedback = True
        scroll_bar = self.feedback_scroll.verticalScrollBar()
        scroll_bar.valueChanged.connect(self.update_feedback_follow)
        scroll_bar.rangeChanged.connect(self.update_feedback_range)
        history.addWidget(self.feedback_scroll, 1)
        content.addWidget(history_card)
        self.setStyleSheet(STYLE)

    def select_video(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Select video", str(ROOT / "videos" / "originales"),
            "Videos (*.mp4 *.mov *.mkv *.avi *.m4v);;All files (*)"
        )
        if selected:
            self.open_source(Path(selected))

    def open_source(self, path: Path) -> None:
        try:
            self.player.load(path)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not open video", str(exc))
            return
        self.source = path
        self.source_height = self.player.frame_height
        self.result = None
        self.reset_feedback()
        self.export_button.setEnabled(False)
        self.process_button.setEnabled(True)
        self.file_label.setText(path.name)
        self.file_label.setToolTip(str(path))
        self.preview_title.setText("Preview · original")
        self.progress.setValue(0)
        self.status.setText("Video ready for analysis")
        if self.work_directory is not None:
            self.work_directory.cleanup()
            self.work_directory = None

    def start_analysis(self) -> None:
        if self.source is None or self.worker is not None:
            return
        self.player.stop()
        self.result = None
        self.reset_feedback()
        self.export_button.setEnabled(False)
        if self.work_directory is not None:
            self.work_directory.cleanup()
        self.work_directory = tempfile.TemporaryDirectory(prefix="crux-analysis-")
        self.worker = AnalysisWorker(self.source, Path(self.work_directory.name), self,
                                     pose_size=self.pose_quality.currentData())
        self.worker.progress_changed.connect(self.update_progress)
        self.worker.completed.connect(self.analysis_complete)
        self.worker.failed.connect(self.analysis_failed)
        self.worker.cancelled.connect(self.analysis_cancelled)
        self.worker.finished.connect(self.worker_finished)
        self.select_button.setEnabled(False)
        self.pose_quality.setEnabled(False)
        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status.setText("Loading existing models…")
        self.worker.start()

    def update_progress(self, stage: str, current: int, total: int) -> None:
        if stage == "load":
            self.status.setText("Loading existing models…")
            self.progress.setRange(0, 0)
        elif stage == "map":
            self.progress.setRange(0, 100)
            self.progress.setValue(round(10 * current / max(1, total)))
            self.status.setText(f"Identifying holds · {current}/{total} frames")
        elif stage == "analyze":
            if total > 0:
                self.progress.setRange(0, 100)
                self.progress.setValue(10 + round(90 * min(current, total) / total))
                self.status.setText(f"Analyzing movement · {current}/{total} frames")
            else:
                self.progress.setRange(0, 0)
                self.status.setText(f"Analyzing movement · {current} frames")

    def cancel_analysis(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status.setText("Cancelling; waiting for the current frame…")

    def analysis_complete(self, video: str, _map: str, frames: int, feedback: list,
                          summary: str) -> None:
        self.result = Path(video)
        self.reset_feedback()
        self.feedback = feedback
        self.overall_recommendation = summary
        self.summary_label.setText(f"Overall recommendation\n{summary}")
        try:
            self.player.load(self.result, display_height=self.source_height,
                             original_path=self.source)
        except (OSError, ValueError) as exc:
            self.result = None
            self.reset_feedback()
            self.analysis_failed(str(exc))
            return
        self.preview_title.setText("Preview · analysis")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status.setText(f"Analysis complete · {frames} frames")
        self.export_button.setEnabled(True)

    def reset_feedback(self) -> None:
        self.feedback = []
        self.overall_recommendation = ""
        self.summary_label.clear()
        self.summary_label.hide()
        self.feedback_cursor = 0
        self.active_feedback.clear()
        while self.feedback_layout.count() > 1:
            item = self.feedback_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.feedback_empty.setText("No feedback yet.")
        self.feedback_empty.show()
        self.follow_feedback = True

    def show_feedback(self, index: int) -> None:
        at_end = (
            bool(self.overall_recommendation)
            and self.player.total_frames > 0
            and index == self.player.total_frames - 1
        )
        self.summary_label.setVisible(at_end)
        self.feedback_empty.setText(
            "No individual feedback was shown." if at_end else "No feedback yet."
        )
        if not self.feedback or index < self.feedback_cursor:
            return
        added = False
        for frame_index in range(self.feedback_cursor, min(index + 1, len(self.feedback))):
            items = [item for item in self.feedback[frame_index] if item[0] != "unknown"]
            current = set(items)
            for kind, text in items:
                if (kind, text) in self.active_feedback:
                    continue
                color = "#{:02x}{:02x}{:02x}".format(*STATUS_COLORS[kind])
                timestamp = self.player.event_timestamp(frame_index / self.player.fps)
                label = QLabel(f"{timestamp}  ·  {STATUS_NAMES[kind]}\n{text}")
                label.setWordWrap(True)
                label.setStyleSheet(
                    f"background: #182232; color: #e8edf5; border: 2px solid {color}; "
                    "border-radius: 7px; padding: 7px 10px;"
                )
                self.feedback_layout.insertWidget(self.feedback_layout.count() - 1, label)
                added = True
            self.active_feedback = current
        self.feedback_cursor = min(index + 1, len(self.feedback))
        if added:
            self.feedback_empty.hide()

    def update_feedback_follow(self, value: int) -> None:
        scroll_bar = self.feedback_scroll.verticalScrollBar()
        self.follow_feedback = value >= scroll_bar.maximum() - 2

    def update_feedback_range(self, _minimum: int, maximum: int) -> None:
        if self.follow_feedback:
            self.feedback_scroll.verticalScrollBar().setValue(maximum)

    def analysis_failed(self, error: str) -> None:
        self.progress.setRange(0, 100)
        self.status.setText("Could not complete the analysis")
        QMessageBox.critical(self, "Analysis error", error)

    def analysis_cancelled(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Analysis cancelled")

    def worker_finished(self) -> None:
        self.worker = None
        self.select_button.setEnabled(True)
        self.pose_quality.setEnabled(True)
        self.process_button.setEnabled(self.source is not None)
        self.cancel_button.setEnabled(False)
        if self.closing:
            self.close()

    def export_result(self) -> None:
        if self.result is None or self.source is None:
            return
        suggested = str(ROOT / "videos" / "analizados" / f"{self.source.stem}_analyzed.mp4")
        selected, _ = QFileDialog.getSaveFileName(self, "Export video", suggested, "MP4 video (*.mp4)")
        if not selected:
            return
        destination = Path(selected)
        if destination.suffix.lower() != ".mp4":
            destination = destination.with_name(destination.name + ".mp4")
        if destination.exists() and QMessageBox.question(
            self, "Replace file", f"{destination.name} already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            export_video(self.result, destination, {
                self.source, ROOT / "models" / "best_holds.pt", ROOT / "models" / "yolov8m-pose.pt"
            })
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not export", str(exc))
            return
        self.status.setText(f"Exported: {destination.name}")
        QMessageBox.information(self, "Video exported", f"Saved to:\n{destination}")

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.closing = True
            self.cancel_analysis()
            event.ignore()
            return
        self.player.stop()
        if self.work_directory is not None:
            self.work_directory.cleanup()
            self.work_directory = None
        event.accept()


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
