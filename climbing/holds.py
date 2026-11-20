"""Map holds on a fixed wall and associate them with climber positions."""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


CLASSES = {"Jug", "Crimp", "Pinch", "Slope", "Pocket", "Volume"}


HOLD_COLORS = {
    "Jug": (85, 195, 255),
    "Crimp": (255, 138, 101),
    "Pinch": (199, 146, 234),
    "Slope": (244, 211, 94),
    "Pocket": (237, 106, 158),
    "Volume": (139, 209, 124),
}


ProgressCallback = Callable[[str, int, int], None]


class ProcessingCancelled(Exception):
    """Raised when the user cancels processing."""


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ProcessingCancelled("Processing cancelled")


@dataclass
class Hold:
    id: int
    category: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]


@dataclass
class Cluster:
    boxes: list[np.ndarray] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    frames: set[int] = field(default_factory=set)

    @property
    def bbox(self) -> np.ndarray:
        return np.mean(self.boxes, axis=0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    lo = np.maximum(a[:2], b[:2])
    hi = np.minimum(a[2:], b[2:])
    intersection = float(np.prod(np.maximum(hi - lo, 0)))
    area_a = float(np.prod(np.maximum(a[2:] - a[:2], 0)))
    area_b = float(np.prod(np.maximum(b[2:] - b[:2], 0)))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def same_hold(a: np.ndarray, b: np.ndarray) -> bool:
    if iou(a, b) >= 0.45:
        return True
    center_distance = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
    min_side = min(*(a[2:] - a[:2]), *(b[2:] - b[:2]))
    return min_side > 0 and center_distance <= 0.2 * min_side


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Cannot open video: {path}")
    return capture


def map_wall(video: Path, model, output: Path, frame_limit: int = 15,
             progress: ProgressCallback | None = None,
             cancelled: Callable[[], bool] | None = None) -> list[Hold]:
    """Cluster detections from the opening frames and save pixel coordinates."""
    capture = open_video(video)
    clusters: list[Cluster] = []
    frames_read = 0
    width = height = 0
    try:
        for frame_index in range(frame_limit):
            check_cancelled(cancelled)
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            height, width = frame.shape[:2]
            result = model.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
            if progress is not None:
                progress("map", frames_read, frame_limit)
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for index in np.argsort(-scores):
                box = boxes[index].astype(float)
                name = str(result.names[int(classes[index])])
                if name not in CLASSES:
                    raise ValueError(f"Unexpected hold class {name!r} in model")
                box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
                if np.any(box[2:] <= box[:2]):
                    continue
                candidates = [
                    cluster for cluster in clusters
                    if frame_index not in cluster.frames and same_hold(cluster.bbox, box)
                ]
                if candidates:
                    cluster = max(candidates, key=lambda item: iou(item.bbox, box))
                else:
                    cluster = Cluster()
                    clusters.append(cluster)
                cluster.boxes.append(box)
                cluster.labels.append(name)
                cluster.frames.add(frame_index)
    finally:
        capture.release()
    check_cancelled(cancelled)
    if not frames_read:
        raise ValueError(f"Video has no readable frames: {video}")
    if not clusters:
        raise ValueError("No holds detected in the opening frames; check the video and weights")
    holds = []
    for cluster in clusters:
        box = cluster.bbox
        center = (box[:2] + box[2:]) / 2
        holds.append(Hold(len(holds) + 1, Counter(cluster.labels).most_common(1)[0][0],
                          tuple(float(v) for v in box), tuple(float(v) for v in center)))
    data = {
        "video": str(video.resolve()),
        "frame_width": width,
        "frame_height": height,
        "frames_mapped": frames_read,
        "holds": [
            {"id": hold.id, "class": hold.category, "bbox": list(hold.bbox),
             "center": list(hold.center), "observations": len(cluster.frames)}
            for hold, cluster in zip(holds, clusters)
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return holds


def near_hold(point: np.ndarray, hold: Hold, padding: float) -> bool:
    x1, y1, x2, y2 = hold.bbox
    return x1 - padding <= point[0] <= x2 + padding and y1 - padding <= point[1] <= y2 + padding


def active_hold(wrist: np.ndarray, torso: float, holds: list[Hold], margin: float) -> Hold | None:
    padding = torso * margin
    candidates = []
    for hold in holds:
        if near_hold(wrist, hold, padding):
            distance = np.linalg.norm(wrist - np.asarray(hold.center)) / torso
            candidates.append((distance, hold.id, hold))
    return min(candidates, key=lambda item: (item[0], item[1]))[2] if candidates else None
