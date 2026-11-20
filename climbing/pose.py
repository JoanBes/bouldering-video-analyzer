"""Track and measure the climber's visible pose."""

import numpy as np


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6


LEFT_ELBOW, RIGHT_ELBOW = 7, 8


LEFT_WRIST, RIGHT_WRIST = 9, 10


LEFT_HIP, RIGHT_HIP = 11, 12


LEFT_KNEE, RIGHT_KNEE = 13, 14


LEFT_ANKLE, RIGHT_ANKLE = 15, 16


SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
    (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)


def angle(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> float | None:
    first, second = a - vertex, c - vertex
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / denominator, -1, 1))))


class PoseSmoother:
    """EMA of visible points, resetting each point after a visibility gap."""

    def __init__(self, alpha: float = 0.55, threshold: float = 0.3):
        self.alpha = alpha
        self.threshold = threshold
        self.previous: np.ndarray | None = None
        self.valid = np.zeros(17, dtype=bool)

    def update(self, points: np.ndarray, confidences: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        valid = (confidences >= self.threshold) & np.isfinite(points).all(axis=1)
        if self.previous is None:
            self.previous = points.copy()
        else:
            persistent = valid & self.valid
            self.previous[persistent] = (
                self.alpha * points[persistent] + (1 - self.alpha) * self.previous[persistent]
            )
            self.previous[valid & ~self.valid] = points[valid & ~self.valid]
        self.valid = valid
        return self.previous.copy(), valid.copy()

    def reset(self) -> None:
        self.previous = None
        self.valid[:] = False


def select_person(result, previous_center: np.ndarray | None, frame_diagonal: float):
    """Select the largest person initially, then a nearby person if present."""
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return None
    confidences = result.keypoints.conf
    if confidences is None:
        return None
    confidence_values = confidences.cpu().numpy()
    valid_people = np.flatnonzero(
        (confidence_values[:, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]] >= 0.3)
        .sum(axis=1) >= 2
    )
    if len(valid_people) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    if previous_center is None:
        index = int(valid_people[np.argmax(areas[valid_people])])
    else:
        distances = np.linalg.norm(centers[valid_people] - previous_center, axis=1)
        nearest = int(np.argmin(distances))
        index = int(valid_people[nearest])
        if distances[nearest] > frame_diagonal * 0.2:
            return None
    return (result.keypoints.xy[index].cpu().numpy(),
            confidence_values[index], centers[index])


def torso_and_hip(points: np.ndarray, valid: np.ndarray):
    if not all(valid[i] for i in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)):
        return None
    shoulders = (points[LEFT_SHOULDER] + points[RIGHT_SHOULDER]) / 2
    hip = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2
    torso = float(np.linalg.norm(shoulders - hip))
    return (torso, hip) if torso > 1e-6 else None


def support_x(points: np.ndarray, valid: np.ndarray, torso: float) -> tuple[float, float] | None:
    ankles = [points[i] for i in (LEFT_ANKLE, RIGHT_ANKLE) if valid[i]]
    if not ankles:
        return None
    lowest = max(point[1] for point in ankles)
    supporting = [point for point in ankles if lowest - point[1] <= 0.25 * torso]
    mean = np.mean(supporting, axis=0)
    return float(mean[0]), float(mean[1])
