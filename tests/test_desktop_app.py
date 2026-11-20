import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel

from app.window import AnalysisWorker, MainWindow, export_video
from climbing_analyzer import HOLD_COLORS, panel_height
from test_climbing_analyzer import FakeHoldModel, FakePoseModel


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def make_video(self, path: Path, frame_count: int = 3):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            writer.write(np.full((128, 128, 3), (index * 20) % 256, dtype=np.uint8))
        writer.release()

    def test_select_seek_and_export_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, result, destination = root / "source.mp4", root / "result.mp4", root / "export.mp4"
            self.make_video(source)
            result.write_bytes(b"annotated")
            window = MainWindow()
            window.open_source(source)
            self.assertEqual(window.source, source)
            self.assertTrue(window.process_button.isEnabled())
            self.assertTrue(window.player.show_frame(2))
            self.assertEqual(window.player.current_frame, 2)
            export_video(result, destination, {source})
            self.assertEqual(destination.read_bytes(), b"annotated")
            with self.assertRaises(ValueError):
                export_video(result, source, {source})
            self.assertNotEqual(source.read_bytes(), b"annotated")
            window.close()

    def test_worker_processes_and_signals_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            self.make_video(source)
            (root / "models").mkdir()
            (root / "models" / "best_holds.pt").write_bytes(b"model")
            (root / "models" / "yolov8m-pose.pt").write_bytes(b"model")
            worker = AnalysisWorker(source, root, pose_size=1024)
            completed, errors, events = [], [], []
            worker.completed.connect(lambda *args: completed.append(args))
            worker.failed.connect(errors.append)
            worker.progress_changed.connect(lambda *args: events.append(args))
            models = [FakeHoldModel(), FakePoseModel()]
            with (patch("app.window.ROOT", root),
                  patch("ultralytics.YOLO", side_effect=models),
                  patch.object(models[1], "predict", wraps=models[1].predict) as predict):
                worker.run()
            self.assertEqual(predict.call_count, 3)
            self.assertTrue(all(call.kwargs["imgsz"] == 1024 for call in predict.call_args_list))
            self.assertEqual(errors, [])
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0][2], 3)
            self.assertTrue(Path(completed[0][0]).is_file())
            self.assertTrue(Path(completed[0][1]).is_file())
            self.assertEqual(len(completed[0][3]), 3)
            self.assertIn("No coaching signals were detected", completed[0][4])
            self.assertTrue(any(event[0] == "map" for event in events))
            self.assertTrue(any(event[0] == "analyze" for event in events))
            window = MainWindow()
            window.open_source(source)
            window.analysis_complete(*completed[0])
            self.assertTrue(window.summary_label.isHidden())
            self.assertEqual(window.player.image.height(), 128)
            self.assertTrue(window.player.overlays.isEnabled())
            window.player.show_frame(1)
            analyzed_image = window.player.image.copy()
            window.player.toggle()
            window.player.overlays.setChecked(False)
            self.assertTrue(window.player.timer.isActive())
            self.assertEqual(window.player.current_frame, 1)
            original_image = window.player.image.copy()
            self.assertNotEqual(analyzed_image, original_image)
            window.player.show_frame(2)
            self.assertFalse(window.summary_label.isHidden())
            self.assertIn(completed[0][4], window.summary_label.text())
            self.assertEqual(window.feedback_empty.text(), "No individual feedback was shown.")
            window.player.overlays.setChecked(True)
            self.assertTrue(window.player.timer.isActive())
            self.assertEqual(window.player.current_frame, 2)
            window.player.show_frame(1)
            self.assertTrue(window.summary_label.isHidden())
            self.assertEqual(window.feedback_empty.text(), "No feedback yet.")
            self.assertEqual(window.player.image, analyzed_image)
            window.player.toggle()
            capture = cv2.VideoCapture(completed[0][0])
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[0], 128 + panel_height(128))
            destination = root / "export.mp4"
            export_video(window.result, destination, {source})
            exported = cv2.VideoCapture(str(destination))
            ok, exported_frame = exported.read()
            exported.release()
            self.assertTrue(ok)
            np.testing.assert_array_equal(exported_frame, frame)
            window.player.show_frame(0)
            window.player.toggle()
            QTest.qWait(350)
            self.assertEqual(window.player.current_frame, 2)
            self.assertFalse(window.player.timer.isActive())
            self.assertFalse(window.summary_label.isHidden())
            window.player.toggle()
            self.assertTrue(window.summary_label.isHidden())
            window.player.toggle()
            window.player.show_frame(2)
            self.assertFalse(window.summary_label.isHidden())
            window.open_source(source)
            self.assertTrue(window.summary_label.isHidden())
            self.assertEqual(window.overall_recommendation, "")
            window.close()

    def test_analysis_quality_selection_is_passed_to_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            self.make_video(source)
            window = MainWindow()
            window.open_source(source)
            self.assertEqual(window.pose_quality.currentData(), 640)
            window.pose_quality.setCurrentIndex(window.pose_quality.findData(1024))
            with patch.object(AnalysisWorker, "start"):
                window.start_analysis()
            self.assertEqual(window.worker.pose_size, 1024)
            self.assertFalse(window.pose_quality.isEnabled())
            window.worker_finished()
            self.assertTrue(window.pose_quality.isEnabled())
            window.close()

    def test_analysis_view_rejects_unsynchronized_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, analyzed = root / "original.mp4", root / "analyzed.mp4"
            self.make_video(original, 3)
            self.make_video(analyzed, 4)
            window = MainWindow()
            with self.assertRaisesRegex(ValueError, "different durations or frame rates"):
                window.player.load(analyzed, original_path=original)
            self.assertIsNone(window.player.capture)
            window.close()

    def test_feedback_history_accumulates_with_timestamps_and_severity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            self.make_video(path, 8)
            window = MainWindow()
            good = ("good", "Active lock-off")
            bad = ("bad", "Check your hips")
            regular = ("regular", "Align your shoulder")
            window.feedback = [
                [("unknown", "No feedback")],
                [good],
                [good],
                [good, bad],
                [bad],
                [("unknown", "No feedback")],
                [bad],
                [regular],
            ]
            window.player.load(path)
            self.assertEqual(window.feedback_layout.count(), 1)
            window.player.show_frame(4)
            self.assertEqual(window.feedback_layout.count(), 3)
            self.assertEqual(window.feedback_layout.itemAt(0).widget().text(),
                             "00:00.100  ·  GOOD\nActive lock-off")
            self.assertEqual(window.feedback_layout.itemAt(1).widget().text(),
                             "00:00.300  ·  NEEDS WORK\nCheck your hips")
            self.assertIn("#f36769", window.feedback_layout.itemAt(1).widget().styleSheet())
            window.player.show_frame(1)
            self.assertEqual(window.feedback_layout.count(), 3)
            window.player.show_frame(7)
            self.assertEqual(window.feedback_layout.count(), 5)
            self.assertEqual(window.feedback_layout.itemAt(2).widget().text(),
                             "00:00.600  ·  NEEDS WORK\nCheck your hips")
            self.assertEqual(window.feedback_layout.itemAt(3).widget().text(),
                             "00:00.700  ·  FAIR\nAlign your shoulder")
            self.assertIn("#f4b947", window.feedback_layout.itemAt(3).widget().styleSheet())
            window.open_source(path)
            self.assertEqual(window.feedback_layout.count(), 1)
            self.assertEqual(window.feedback, [])
            window.close()

    def test_playback_speed_changes_timer_during_playback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            self.make_video(path, 20)
            window = MainWindow()
            window.player.load(path)
            speed = window.player.speed
            speed.setCurrentIndex(speed.findData(0.5))
            window.player.toggle()
            self.assertEqual(window.player.timer.interval(), 200)
            speed.setCurrentIndex(speed.findData(2.0))
            self.assertEqual(window.player.timer.interval(), 50)
            QTest.qWait(130)
            self.assertGreaterEqual(window.player.current_frame, 1)
            window.player.toggle()
            self.assertFalse(window.player.timer.isActive())
            window.player.show_frame(10)
            self.assertEqual(window.player.current_frame, 10)
            window.close()

    def test_history_scrolls_and_respects_manual_scroll(self):
        window = MainWindow()
        window.show()
        window.player.fps = 10
        window.feedback = [[("bad", f"Cue {index}")] for index in range(50)]
        window.show_feedback(39)
        QTest.qWait(50)
        scroll = window.feedback_scroll.verticalScrollBar()
        self.assertGreater(scroll.maximum(), 0)
        self.assertEqual(scroll.value(), scroll.maximum())
        scroll.setValue(0)
        window.show_feedback(49)
        QTest.qWait(50)
        self.assertEqual(scroll.value(), 0)
        self.assertEqual(window.feedback_layout.count(), 51)
        window.close()

    def test_english_controls_and_hold_color_legend(self):
        window = MainWindow()
        self.assertEqual(window.windowTitle(), "Crux · Climbing Analysis")
        self.assertEqual(window.select_button.text(), "Open video")
        self.assertEqual(window.process_button.text(), "Start analysis")
        self.assertEqual(window.export_button.text(), "Export video")
        labels = [label.text() for label in window.findChildren(QLabel)]
        self.assertIn("Hold box colors", labels)
        for category, color in HOLD_COLORS.items():
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            self.assertTrue(any(hex_color in label and category in label for label in labels))
        window.close()


if __name__ == "__main__":
    unittest.main()
