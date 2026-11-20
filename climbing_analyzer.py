"""Map a fixed climbing wall and render confidence-gated 2D coaching heuristics."""

import argparse
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

# Re-export the original public API for existing callers.
from climbing.coaching import (
    Coach, CoachingSummary, CueFilter, feedback_items,
    FEEDBACK as FEEDBACK, STATUS_COLORS as STATUS_COLORS, STATUS_NAMES as STATUS_NAMES,
    select_key_cue as select_key_cue,
)
from climbing.holds import (
    ROOT, Hold, ProgressCallback, active_hold, check_cancelled, map_wall, open_video,
    CLASSES as CLASSES, HOLD_COLORS as HOLD_COLORS, Cluster as Cluster,
    ProcessingCancelled as ProcessingCancelled, iou as iou, near_hold as near_hold,
    same_hold as same_hold,
)
from climbing.pose import (
    LEFT_ANKLE, LEFT_WRIST, RIGHT_ANKLE, RIGHT_WRIST,
    PoseSmoother, angle, select_person, support_x, torso_and_hip,
    LEFT_ELBOW as LEFT_ELBOW, LEFT_HIP as LEFT_HIP, LEFT_KNEE as LEFT_KNEE,
    LEFT_SHOULDER as LEFT_SHOULDER, RIGHT_ELBOW as RIGHT_ELBOW,
    RIGHT_HIP as RIGHT_HIP, RIGHT_KNEE as RIGHT_KNEE, RIGHT_SHOULDER as RIGHT_SHOULDER,
    SKELETON as SKELETON,
)
from climbing.rendering import draw_feedback_frame, draw_frame, pad_to_even, panel_height


def analyze_video(video: Path, output: Path, holds: list[Hold], pose_model,
                  wrist_margin: float, progress: ProgressCallback | None = None,
                  cancelled: Callable[[], bool] | None = None,
                  enhanced: bool = False,
                  feedback_callback: Callable[[list[tuple[str, str]]], None] | None = None,
                  pose_size: int = 640,
                  summary_callback: Callable[[str], None] | None = None) -> int:
    if pose_size not in (640, 1024):
        raise ValueError("Pose image size must be 640 or 1024")
    capture = open_video(video)
    writer = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video must have a valid FPS and frame dimensions")
        output.parent.mkdir(parents=True, exist_ok=True)
        output_height = height + panel_height(width) if enhanced else height
        output_width = width + width % 2 if enhanced else width
        if enhanced:
            output_height += output_height % 2
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (output_width, output_height))
        if not writer.isOpened():
            raise OSError(f"Cannot create output video: {output}")
        smoother, coach, cue_filter = PoseSmoother(), Coach(), CueFilter()
        summary = CoachingSummary()
        center = None
        missing = 0
        processed = 0
        with tqdm(total=count if count > 0 else None, unit="frame", desc="Analyzing",
                  disable=progress is not None) as progress_bar:
            while True:
                check_cancelled(cancelled)
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError("Video frame dimensions changed during processing")
                result = pose_model.predict(frame, imgsz=pose_size, conf=0.20, verbose=False)[0]
                selected = select_person(result, center, math.hypot(width, height))
                points, valid = None, None
                metrics, cues = [], []
                has_grip = False
                highlighted_holds: list[Hold] = []
                if selected is None:
                    missing += 1
                    smoother.reset()
                    coach.reset()
                    if missing > max(1, round(fps * 0.5)):
                        center = None
                    metrics.append("Climber pose unavailable")
                else:
                    missing = 0
                    raw, confidence, center = selected
                    points, valid = smoother.update(raw, confidence)
                    angles = {}
                    for name, indices in (
                        ("left elbow", (5, 7, 9)), ("right elbow", (6, 8, 10)),
                        ("left knee", (11, 13, 15)), ("right knee", (12, 14, 16)),
                    ):
                        angles[name] = angle(*(points[i] for i in indices)) if all(valid[i] for i in indices) else None
                    metrics.append(" | ".join(
                        f"{name}: {value:.0f} deg" if value is not None else f"{name}: --"
                        for name, value in angles.items()
                    ))
                    body = torso_and_hip(points, valid)
                    if body is None:
                        coach.reset()
                        metrics.append("Torso or hip proxy unavailable; grip and balance disabled")
                    else:
                        torso, hip = body
                        support = support_x(points, valid, torso)
                        grips = {
                            side: active_hold(points[index], torso, holds, wrist_margin) if valid[index] else None
                            for side, index in (("left", LEFT_WRIST), ("right", RIGHT_WRIST))
                        }
                        has_grip = any(hold is not None for hold in grips.values())
                        metrics.append(" | ".join(
                            f"{side} grip: {hold.category} #{hold.id}" if hold else f"{side} grip: --"
                            for side, hold in grips.items()
                        ))
                        if support is not None:
                            offset = (float(hip[0]) - support[0]) / torso
                            metrics.append(f"Hip-support x / torso: {offset:+.2f}")
                            cv2.circle(frame, tuple(np.rint(hip).astype(int)), 5, (255, 100, 0), -1)
                        else:
                            metrics.append("Hip-support offset: --")
                        ankles = [points[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if valid[i]]
                        cues = coach.evaluate(grips, angles, hip, torso, support, processed / fps, ankles)
                summary.update(cues, has_grip)
                cues = cue_filter.update(cues, processed / fps, coach.cue_holds)
                if cues:
                    highlighted_holds = coach.cue_holds[cues[0]]
                if enhanced:
                    frame = pad_to_even(draw_feedback_frame(
                        frame, holds, points, valid, metrics, cues, highlighted_holds
                    ))
                else:
                    draw_frame(frame, holds, points, valid, metrics, cues,
                               highlighted_holds=highlighted_holds)
                writer.write(frame)
                processed += 1
                progress_bar.update(1)
                if feedback_callback is not None:
                    feedback_callback(feedback_items(cues, coach.cue_holds))
                if progress is not None:
                    progress("analyze", processed, count)
        check_cancelled(cancelled)
        if processed == 0:
            raise ValueError("No video frames were processed")
        if count > 0 and processed < count:
            raise OSError(f"Video ended or failed to decode after {processed} of {count} reported frames")
        if summary_callback is not None:
            summary_callback(summary.recommendation())
        return processed
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "videos" / "originales" / "climbing.mp4")
    parser.add_argument("--holds", type=Path, default=ROOT / "models" / "best_holds.pt")
    parser.add_argument("--pose", default=str(ROOT / "models" / "yolov8m-pose.pt"))
    parser.add_argument("--pose-size", type=int, choices=(640, 1024), default=640,
                        help="Pose inference size: 640 is faster; 1024 may detect more detail")
    parser.add_argument("--map", type=Path, default=ROOT / "videos" / "analizados" / "wall_map.json")
    parser.add_argument("--output", type=Path, default=ROOT / "videos" / "analizados" / "climbing_analyzed.mp4")
    parser.add_argument("--wrist-margin", type=float, default=0.15,
                        help="Wrist-to-box tolerance as a fraction of torso length")
    args = parser.parse_args()
    if not math.isfinite(args.wrist_margin) or args.wrist_margin < 0:
        parser.error("--wrist-margin must be finite and non-negative")
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    if not args.holds.is_file():
        parser.error(f"Hold weights not found: {args.holds}")
    paths = [args.video.resolve(), args.holds.resolve(), args.map.resolve(), args.output.resolve()]
    if Path(args.pose).is_file():
        paths.append(Path(args.pose).resolve())
    if len(set(paths)) != len(paths):
        parser.error("Video, model weights, map, and output must be distinct paths")
    from ultralytics import YOLO

    holds = map_wall(args.video, YOLO(str(args.holds)), args.map)
    print(f"Mapped {len(holds)} holds to {args.map}")
    frames = analyze_video(args.video, args.output, holds, YOLO(args.pose), args.wrist_margin,
                           pose_size=args.pose_size,
                           summary_callback=lambda text: print(f"Overall recommendation: {text}"))
    print(f"Saved {frames} annotated frames to {args.output}")


if __name__ == "__main__":
    main()
