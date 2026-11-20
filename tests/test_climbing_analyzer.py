import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from climbing.coaching import CoachingSummary
from climbing_analyzer import (
    CLASSES, HOLD_COLORS, STATUS_NAMES, Coach, CueFilter, Hold, PoseSmoother, ProcessingCancelled,
    active_hold, analyze_video, angle, draw_feedback_frame, draw_frame, feedback_items,
    map_wall, pad_to_even, panel_height, select_person, support_x,
)


def hold(category="Jug"):
    return Hold(1, category, (20.0, 20.0, 40.0, 40.0), (30.0, 30.0))


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeHoldModel:
    def __init__(self):
        self.calls = 0

    def predict(self, frame, **kwargs):
        self.calls += 1
        boxes = type("Boxes", (), {
            "xyxy": Tensor([[20, 20, 40, 40], [55, 20, 75, 40]]),
            "cls": Tensor([0, 1]),
            "conf": Tensor([0.9, 0.8]),
        })()
        return [type("Result", (), {"boxes": boxes, "names": {0: "Jug", 1: "Crimp"}})()]


class FakePoseModel:
    def predict(self, frame, **kwargs):
        boxes = type("Boxes", (), {
            "xyxy": Tensor([[15, 10, 90, 110]]),
            "__len__": lambda self: 1,
        })()
        points = np.zeros((1, 17, 2), dtype=float)
        points[0, 5:17] = [
            [35, 40], [65, 40], [35, 30], [65, 30], [30, 30], [65, 20],
            [40, 65], [60, 65], [40, 80], [60, 80], [40, 100], [60, 100],
        ]
        keypoints = type("Keypoints", (), {
            "xy": Tensor(points), "conf": Tensor(np.ones((1, 17))),
        })()
        # Tensor indexing mirrors the subset of the torch API used by the analyzer.
        return [type("Result", (), {"boxes": boxes, "keypoints": keypoints})()]


Tensor.__getitem__ = lambda self, index: Tensor(self.value[index])


class AnalyzerTests(unittest.TestCase):
    def test_smoothing_gaps_and_angles(self):
        smooth = PoseSmoother()
        points = np.ones((17, 2)) * 10
        confidence = np.ones(17)
        smooth.update(points, confidence)
        result, _ = smooth.update(points + 10, confidence)
        self.assertAlmostEqual(result[0, 0], 15.5)
        confidence[0] = 0
        smooth.update(points, confidence)
        confidence[0] = 1
        result, _ = smooth.update(points + 20, confidence)
        self.assertEqual(result[0, 0], 30)
        self.assertAlmostEqual(angle(np.array([1, 0]), np.array([0, 0]), np.array([0, 1])), 90)
        self.assertIsNone(angle(np.array([0, 0]), np.array([0, 0]), np.array([0, 1])))

    def test_hold_margin_and_foot_proxy(self):
        self.assertIsNone(active_hold(np.array([43, 30]), 100, [hold()], 0.02))
        self.assertEqual(active_hold(np.array([43, 30]), 100, [hold()], 0.03).id, 1)
        points = np.zeros((17, 2))
        points[15], points[16] = [10, 100], [80, 60]
        valid = np.zeros(17, dtype=bool)
        valid[15:17] = True
        self.assertEqual(support_x(points, valid, 100), (10.0, 100.0))

    def test_tracker_ignores_detections_without_visible_torso(self):
        boxes = type("Boxes", (), {
            "xyxy": Tensor([[0, 0, 120, 120], [20, 20, 60, 100]]),
            "__len__": lambda self: 2,
        })()
        points = np.zeros((2, 17, 2))
        confidence = np.zeros((2, 17))
        confidence[1, [5, 6, 11, 12]] = 0.9
        keypoints = type("Keypoints", (), {
            "xy": Tensor(points), "conf": Tensor(confidence),
        })()
        result = type("Result", (), {"boxes": boxes, "keypoints": keypoints})()
        selected = select_person(result, None, 200)
        self.assertIsNotNone(selected)
        np.testing.assert_array_equal(selected[2], [40, 60])
        confidence[1] = 0
        self.assertIsNone(select_person(result, None, 200))

    def test_coaching_rules_and_missing_data(self):
        coach = Coach()
        hip = np.array([0.0, 0.0])
        angles = {"left elbow": 100.0}
        def evaluate(category, time, hip_pos=hip, support=(0, 100), ankles=None):
            return coach.evaluate({"left": hold(category)}, angles, hip_pos, 100, support,
                                  time, ankles if ankles is not None else [np.array([30, 30])])
        self.assertEqual(evaluate("Jug", 0), [])
        self.assertEqual(evaluate("Jug", 1.49), [])
        self.assertIn("Straighten", evaluate("Jug", 1.5)[0])
        coach.reset()
        evaluate("Jug", 0)
        self.assertEqual(evaluate("Jug", 1.5, np.array([0.0, -11.0])), [])
        self.assertEqual(evaluate("Crimp", 2), [])
        self.assertTrue(any("off-center" in cue for cue in evaluate("Crimp", 2.1, support=(50, 100))))
        self.assertTrue(any("Lower" in cue for cue in evaluate("Slope", 2.2, support=(50, 100))))
        self.assertEqual(evaluate("Pinch", 2.3, support=None), [])
        self.assertEqual(evaluate("Pinch", 2.4), [])
        self.assertTrue(any("check your balance" in cue for cue in evaluate("Pinch", 2.5, support=(50, 100))))
        coach.reset()
        evaluate("Pocket", 0)
        evaluate("Pocket", 0.1)
        self.assertTrue(any("jerks" in cue for cue in evaluate("Pocket", 0.2, np.array([20.0, 0]))))
        self.assertTrue(any("off-center" in cue for cue in evaluate("Volume", 0.3, support=(50, 100))))
        self.assertFalse(any("No visible foot" in cue for cue in evaluate("Volume", 0.4, ankles=[np.array([90, 90])])))
        self.assertTrue(any("No visible foot" in cue for cue in evaluate(
            "Volume", 0.5, ankles=[np.array([90, 90]), np.array([95, 95])])))
        self.assertEqual(evaluate("Slope", 0.6, support=None), [])

    def test_key_cues_are_prioritized_stable_and_not_repeated_immediately(self):
        cue_filter = CueFilter()
        positive = "Crimp: Maintain active lock-off and keep hips close"
        caution = "Pinch: Align your shoulder with the direction of pull"
        negative = "Crimp: Hips are off-center from supporting feet"
        self.assertEqual(cue_filter.update([positive, caution, negative], 0), [])
        self.assertEqual(cue_filter.update([positive, caution, negative], 0.49), [])
        self.assertEqual(cue_filter.update([positive, caution, negative], 0.5), [negative])
        self.assertEqual(cue_filter.update([negative], 1), [negative])
        self.assertEqual(cue_filter.update([], 1.1), [])
        self.assertEqual(cue_filter.update([negative], 1.2), [])
        self.assertEqual(cue_filter.update([negative], 2), [])
        self.assertEqual(cue_filter.update([negative], 6), [negative])

    def test_suppressed_cue_does_not_block_another_persistent_cue(self):
        cue_filter = CueFilter()
        negative = "Crimp: Hips are off-center from supporting feet"
        caution = "Pinch: Align your shoulder with the direction of pull"
        self.assertEqual(cue_filter.update([negative], 0), [])
        self.assertEqual(cue_filter.update([negative], 0.5), [negative])
        self.assertEqual(cue_filter.update([], 1), [])
        self.assertEqual(cue_filter.update([negative, caution], 1.1), [])
        self.assertEqual(cue_filter.update([negative, caution], 1.6), [caution])
        self.assertEqual(cue_filter.update([negative, caution], 5.5), [negative])

    def test_cue_stability_and_cooldown_are_specific_to_the_hold(self):
        cue_filter = CueFilter()
        cue = "Crimp: Hips are off-center from supporting feet"
        first = hold("Crimp")
        second = Hold(2, "Crimp", (50, 20, 70, 40), (60, 30))
        first_map, second_map = {cue: [first]}, {cue: [second]}
        self.assertEqual(cue_filter.update([cue], 0, first_map), [])
        self.assertEqual(cue_filter.update([cue], 0.5, first_map), [cue])
        self.assertEqual(cue_filter.update([cue], 0.6, second_map), [])
        self.assertEqual(cue_filter.update([cue], 1.1, second_map), [cue])
        self.assertEqual(feedback_items([cue], second_map), [("bad", f"{cue} (hold #2)")])
        self.assertEqual(feedback_items([cue], {cue: [second, first, first]}),
                         [("bad", f"{cue} (holds #1, #2)")])
        self.assertEqual(cue_filter.update([cue], 1.2, first_map), [])
        self.assertEqual(cue_filter.update([cue], 1.8, first_map), [])

    def test_summary_ranks_topics_without_double_counting_simultaneous_cues(self):
        summary = CoachingSummary()
        crimp = "Crimp: Hips are off-center from supporting feet"
        pinch = "Pinch: Hips appear offset from visible feet; check your balance"
        pocket = "Pocket: Controlled movement, avoid dynamic jerks"
        summary.update([crimp, pinch], True)
        summary.update([pocket], True)
        summary.update([pocket], True)
        result = summary.recommendation()
        self.assertIn("Main focus: Review abrupt hip movements", result)
        self.assertIn("Other signals to review: balance", result)
        self.assertIn("unfiltered 2D estimates", result)

    def test_summary_distinguishes_no_data_from_no_detected_signals(self):
        summary = CoachingSummary()
        summary.update([], False)
        self.assertIn("Not enough visible pose", summary.recommendation())
        summary.update([], True)
        self.assertIn("No coaching signals were detected", summary.recommendation())
        self.assertIn("does not establish correct technique", summary.recommendation())

    def test_summary_includes_raw_cues_that_never_reach_the_video(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.mp4"
            output = Path(directory) / "out.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()

            class OffsetPoseModel(FakePoseModel):
                def predict(self, frame, **kwargs):
                    result = super().predict(frame, **kwargs)
                    result[0].keypoints.xy.value[0, 15:17, 0] = 100
                    return result

            feedback, summaries = [], []
            analyze_video(source, output, [hold("Volume")], OffsetPoseModel(), 0.15,
                          feedback_callback=feedback.append, summary_callback=summaries.append)
            self.assertTrue(all(items[0][0] == "unknown" for items in feedback))
            self.assertEqual(len(summaries), 1)
            self.assertIn("Main focus: Review your hip position", summaries[0])
            self.assertIn("Other signals to review: foot placement", summaries[0])

    def test_recommendation_identifies_and_circles_its_holds(self):
        coach = Coach()
        first = Hold(1, "Crimp", (20, 20, 40, 40), (30, 30))
        second = Hold(2, "Pinch", (70, 40, 90, 60), (80, 50))
        cues = coach.evaluate({"left": first, "right": second}, {"left elbow": 90},
                              np.array([0.0, 0.0]), 100, (50, 100), 0, [])
        negative = "Crimp: Hips are off-center from supporting feet"
        self.assertIn(negative, cues)
        self.assertEqual(coach.cue_holds[negative], [first])
        self.assertEqual(coach.cue_holds["Pinch: Hips appear offset from visible feet; check your balance"],
                         [second])
        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        with patch("climbing_analyzer.cv2.circle", wraps=cv2.circle) as circle:
            draw_frame(frame, [first, second], None, None, [], [negative],
                       highlighted_holds=coach.cue_holds[negative])
        red_circles = [call for call in circle.call_args_list if call.args[3] == (0, 0, 255)]
        self.assertEqual(len(red_circles), 1)
        self.assertEqual(red_circles[0].args[1], (30, 30))
        self.assertGreater(red_circles[0].args[2], 14)

    def test_analysis_renders_circle_only_when_feedback_is_shown(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "in.mp4"
            output = Path(directory) / "out.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(10):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()
            pinch = hold("Pinch")
            feedback = []
            class OffsetPoseModel(FakePoseModel):
                def predict(self, frame, **kwargs):
                    result = super().predict(frame, **kwargs)
                    result[0].keypoints.xy.value[0, 15:17, 0] = 100
                    return result
            with patch("climbing_analyzer.draw_feedback_frame", wraps=draw_feedback_frame) as draw:
                analyze_video(source, output, [pinch], OffsetPoseModel(), 0.15,
                              enhanced=True, feedback_callback=feedback.append)
            self.assertEqual(feedback[4][0][0], "unknown")
            self.assertEqual(feedback[5], [
                ("bad", "Pinch: Hips appear offset from visible feet; check your balance (hold #1)")
            ])
            self.assertEqual(draw.call_args_list[4].args[6], [])
            self.assertEqual(draw.call_args_list[5].args[6], [pinch])
            capture = cv2.VideoCapture(str(output))
            capture.set(cv2.CAP_PROP_POS_FRAMES, 5)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            region = frame[46:53, 26:35]
            self.assertTrue(np.any((region[:, :, 2] > 170) & (region[:, :, 1] < 100)
                                   & (region[:, :, 0] < 100)))

    def test_mapping_and_full_video_render(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source, mapped, output = path / "in.mp4", path / "map.json", path / "out.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(20):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()
            model = FakeHoldModel()
            holds = map_wall(source, model, mapped)
            data = json.loads(mapped.read_text())
            self.assertEqual(model.calls, 15)
            self.assertEqual(len(holds), 2)
            self.assertEqual(data["frames_mapped"], 15)
            self.assertEqual(data["holds"][0]["observations"], 15)
            self.assertEqual(data["holds"][1]["class"], "Crimp")
            self.assertEqual(analyze_video(source, output, holds, FakePoseModel(), 0.15), 20)
            capture = cv2.VideoCapture(str(output))
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 20)
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertGreater(frame.sum(), 0)

    def test_feedback_colors_and_no_false_positive(self):
        unknown = feedback_items([])
        self.assertEqual(unknown[0][0], "unknown")
        self.assertEqual(unknown[0][1], "No feedback available for this frame.")
        self.assertEqual(STATUS_NAMES,
                         {"good": "GOOD", "regular": "FAIR", "bad": "NEEDS WORK",
                          "unknown": "NO ASSESSMENT"})
        positive = "Crimp: Maintain active lock-off and keep hips close"
        caution = "Pinch: Align your shoulder with the direction of pull"
        negative = "Pocket: Controlled movement, avoid dynamic jerks"
        self.assertEqual(feedback_items([positive, caution, negative]),
                         [("good", positive), ("regular", caution), ("bad", negative)])
        rendered = draw_feedback_frame(np.zeros((400, 480, 3), dtype=np.uint8),
                                       [], None, None, [], [positive, caution, negative])
        self.assertEqual(rendered.shape, (400 + panel_height(480), 480, 3))
        self.assertTrue(np.any(np.all(rendered[400:] == [148, 204, 57], axis=2)))
        self.assertTrue(np.any(np.all(rendered[400:] == [71, 185, 244], axis=2)))
        self.assertTrue(np.any(np.all(rendered[400:] == [105, 103, 243], axis=2)))
        odd = np.ones((101, 127, 3), dtype=np.uint8)
        padded = pad_to_even(odd)
        self.assertEqual(padded.shape, (102, 128, 3))
        np.testing.assert_array_equal(padded[:101, :127], odd)

    def test_hold_boxes_and_labels_use_distinct_class_colors(self):
        self.assertEqual(set(HOLD_COLORS), CLASSES)
        self.assertEqual(len(set(HOLD_COLORS.values())), len(CLASSES))
        holds = [
            Hold(i, category, (10 + i * 75, 40, 50 + i * 75, 65), (30 + i * 75, 52))
            for i, category in enumerate(HOLD_COLORS)
        ]
        for subtle in (False, True):
            with self.subTest(subtle=subtle):
                frame = np.zeros((100, 500, 3), dtype=np.uint8)
                with patch("climbing_analyzer.cv2.putText", wraps=cv2.putText) as put_text:
                    draw_frame(frame, holds, None, None, [], [], subtle_holds=subtle)
                for i, hold_item in enumerate(holds):
                    bgr = tuple(reversed(HOLD_COLORS[hold_item.category]))
                    expected = np.rint(np.array(bgr) * (0.45 if subtle else 1)).astype(np.uint8)
                    np.testing.assert_array_equal(frame[52, 10 + i * 75], expected)
                    self.assertEqual(put_text.call_args_list[i].args[1],
                                     f"{hold_item.id}:{hold_item.category}")
                    self.assertEqual(put_text.call_args_list[i].args[5], bgr)

    def test_enhanced_video_progress_and_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source, mapped, output = path / "in.mp4", path / "map.json", path / "out.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()
            events = []
            holds = map_wall(source, FakeHoldModel(), mapped,
                             progress=lambda *args: events.append(args))
            self.assertEqual(events[-1], ("map", 3, 15))
            events.clear()
            pose_model = FakePoseModel()
            with patch.object(pose_model, "predict", wraps=pose_model.predict) as predict:
                self.assertEqual(analyze_video(source, output, holds, pose_model, 0.15,
                                               progress=lambda *args: events.append(args),
                                               enhanced=True), 3)
            self.assertEqual(predict.call_count, 3)
            self.assertTrue(all(call.kwargs["imgsz"] == 640 for call in predict.call_args_list))
            with self.assertRaisesRegex(ValueError, "640 or 1024"):
                analyze_video(source, output, holds, pose_model, 0.15, pose_size=320)
            self.assertEqual(events[-1], ("analyze", 3, 3))
            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertEqual(frame.shape[:2], (128 + panel_height(128), 128))
            with self.assertRaises(ProcessingCancelled):
                map_wall(source, FakeHoldModel(), path / "cancel.json", cancelled=lambda: True)
            self.assertFalse((path / "cancel.json").exists())
            summaries = []
            with self.assertRaises(ProcessingCancelled):
                analyze_video(source, output, holds, pose_model, 0.15,
                              cancelled=lambda: True, summary_callback=summaries.append)
            self.assertEqual(summaries, [])

if __name__ == "__main__":
    unittest.main()
