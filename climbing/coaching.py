"""Generate and filter illustrative 2D coaching cues."""

import numpy as np

from .holds import Hold, near_hold


class Coach:
    def __init__(self):
        self.jug: dict[str, tuple[int, float, float]] = {}
        self.cue_holds: dict[str, list[Hold]] = {}
        self.previous_hip: np.ndarray | None = None
        self.previous_velocity: np.ndarray | None = None
        self.previous_time: float | None = None

    def reset(self) -> None:
        self.jug.clear()
        self.cue_holds.clear()
        self.previous_hip = None
        self.previous_velocity = None
        self.previous_time = None

    def evaluate(self, grips: dict[str, Hold | None], angles: dict[str, float | None],
                 hip: np.ndarray, torso: float, support: tuple[float, float] | None,
                 timestamp: float, ankles: list[np.ndarray]) -> list[str]:
        acceleration = None
        if self.previous_time is not None and self.previous_hip is not None:
            dt = timestamp - self.previous_time
            if 0 < dt <= 0.5:
                velocity = (hip - self.previous_hip) / dt
                if self.previous_velocity is not None:
                    acceleration = float(np.linalg.norm(velocity - self.previous_velocity) / dt / torso)
                self.previous_velocity = velocity
            else:
                self.previous_velocity = None
        self.previous_hip = hip.copy()
        self.previous_time = timestamp
        offset = abs(float(hip[0]) - support[0]) / torso if support else None
        cues = []
        self.cue_holds.clear()
        for side, hold in grips.items():
            previous_cue_count = len(cues)
            elbow = angles.get(f"{side} elbow")
            if hold is None or hold.category != "Jug" or elbow is None or elbow >= 120:
                self.jug.pop(side, None)
            if hold is None:
                continue
            if hold.category == "Jug" and elbow is not None and elbow < 120:
                state = self.jug.get(side)
                if state is None or state[0] != hold.id or state[2] - hip[1] > 0.1 * torso:
                    state = (hold.id, timestamp, float(hip[1]))
                    self.jug[side] = state
                if timestamp - state[1] >= 1.5:
                    cues.append("Jug detected: Straighten your arms to rest and save energy")
            elif hold.category == "Crimp":
                if offset is not None and offset > 0.35:
                    cues.append("Crimp: Hips are off-center from supporting feet")
            elif hold.category == "Slope":
                if support is not None and (offset > 0.35 or (support[1] - hip[1]) / torso > 1.25):
                    cues.append("Slope: Lower your center of gravity and maintain body tension")
            elif hold.category == "Pinch":
                if offset is not None and offset > 0.35:
                    cues.append("Pinch: Hips appear offset from visible feet; check your balance")
            elif hold.category == "Pocket":
                if acceleration is not None and acceleration > 4.0:
                    cues.append("Pocket: Controlled movement, avoid dynamic jerks")
            elif hold.category == "Volume" and support is not None:
                if offset > 0.35:
                    cues.append("Volume: Hips are off-center from supporting feet")
                if len(ankles) == 2 and not any(
                    near_hold(ankle, hold, 0.15 * torso) for ankle in ankles
                ):
                    cues.append("Volume: No visible foot near this volume")
            for cue in cues[previous_cue_count:]:
                self.cue_holds.setdefault(cue, []).append(hold)
        return list(dict.fromkeys(cues))


FEEDBACK = {
    "Jug detected: Straighten your arms to rest and save energy": "bad",
    "Crimp: Maintain active lock-off and keep hips close": "good",
    "Crimp: Hips are off-center from supporting feet": "bad",
    "Slope: Lower your center of gravity and maintain body tension": "bad",
    "Pinch: Align your shoulder with the direction of pull": "regular",
    "Pinch: Hips appear offset from visible feet; check your balance": "bad",
    "Pocket: Controlled movement, avoid dynamic jerks": "bad",
    "Volume: Trust foot friction and center your hips": "regular",
    "Volume: Hips are off-center from supporting feet": "bad",
    "Volume: No visible foot near this volume": "regular",
}


STATUS_COLORS = {
    "good": (57, 204, 148),
    "regular": (244, 185, 71),
    "bad": (243, 103, 105),
    "unknown": (135, 148, 166),
}


STATUS_NAMES = {"good": "GOOD", "regular": "FAIR", "bad": "NEEDS WORK",
                "unknown": "NO ASSESSMENT"}

_CUE_PRIORITY = {"bad": 0, "regular": 1, "good": 2}

_SUMMARY_TOPICS = {
    "balance": ("Balance", "Review your hip position relative to your visible feet and "
                "practice maintaining body tension."),
    "rest": ("Resting position", "When stable, consider resting with straighter arms on larger holds."),
    "control": ("Movement control", "Review abrupt hip movements and practice moving with control."),
    "feet": ("Foot placement", "Review your foot placement near volumes and consider available footholds."),
}
_CUE_TOPICS = {
    "Jug detected: Straighten your arms to rest and save energy": "rest",
    "Crimp: Hips are off-center from supporting feet": "balance",
    "Slope: Lower your center of gravity and maintain body tension": "balance",
    "Pinch: Align your shoulder with the direction of pull": "balance",
    "Pinch: Hips appear offset from visible feet; check your balance": "balance",
    "Pocket: Controlled movement, avoid dynamic jerks": "control",
    "Volume: Trust foot friction and center your hips": "feet",
    "Volume: Hips are off-center from supporting feet": "balance",
    "Volume: No visible foot near this volume": "feet",
}


class CoachingSummary:
    """Aggregate unfiltered coaching signals once per topic and video frame."""

    def __init__(self) -> None:
        self.frames: dict[str, int] = {topic: 0 for topic in _SUMMARY_TOPICS}
        self.frames_with_grip = 0

    def update(self, cues: list[str], has_grip: bool) -> None:
        if has_grip:
            self.frames_with_grip += 1
        topics = {_CUE_TOPICS[cue] for cue in cues if FEEDBACK[cue] != "good"}
        for topic in topics:
            self.frames[topic] += 1

    def recommendation(self) -> str:
        topics = sorted(
            (topic for topic, count in self.frames.items() if count),
            key=lambda topic: -self.frames[topic],
        )
        if not topics:
            if not self.frames_with_grip:
                return ("Not enough visible pose and possible hold associations to make a recommendation. "
                        "Try a clear, fixed-camera recording.")
            return ("No coaching signals were detected where a pose and possible hold were visible. "
                    "This does not establish correct technique.")
        main = _SUMMARY_TOPICS[topics[0]][1]
        others = ", ".join(_SUMMARY_TOPICS[topic][0].lower() for topic in topics[1:])
        detail = f" Other signals to review: {others}." if others else ""
        return (f"Main focus: {main}{detail} "
                "Based on unfiltered 2D estimates, which may be noisy.")


def select_key_cue(cues: list[str]) -> list[str]:
    """Keep the most actionable cue, with stable ordering within each severity."""
    return [min(cues, key=lambda cue: _CUE_PRIORITY[FEEDBACK[cue]])] if cues else []


class CueFilter:
    """Track persistent cues independently, including their associated holds."""

    def __init__(self) -> None:
        self.since: dict[tuple[str, tuple[int, ...]], float] = {}
        self.active: tuple[str, tuple[int, ...]] | None = None
        self.last_shown: dict[tuple[str, tuple[int, ...]], float] = {}

    def update(self, cues: list[str], timestamp: float,
               cue_holds: dict[str, list[Hold]] | None = None) -> list[str]:
        current = [
            (cue, tuple(sorted({hold.id for hold in cue_holds.get(cue, [])})))
            if cue_holds is not None else (cue, ())
            for cue in cues
        ]
        current_set = set(current)
        self.since = {key: start for key, start in self.since.items() if key in current_set}
        for key in current:
            self.since.setdefault(key, timestamp)
        eligible = [
            key for key in current
            if timestamp - self.since[key] >= 0.5
            and (key == self.active or key not in self.last_shown
                 or timestamp - self.last_shown[key] >= 5.0)
        ]
        if not eligible:
            self.active = None
            return []
        selected = min(eligible, key=lambda key: (_CUE_PRIORITY[FEEDBACK[key[0]]], key != self.active))
        self.active = selected
        self.last_shown[selected] = timestamp
        return [selected[0]]


def feedback_items(cues: list[str],
                   cue_holds: dict[str, list[Hold]] | None = None) -> list[tuple[str, str]]:
    """Classify rule outputs without treating absence of advice as good technique."""
    if not cues:
        return [("unknown", "No feedback available for this frame.")]
    items = []
    for cue in cues:
        hold_ids = sorted({hold.id for hold in cue_holds.get(cue, [])}) if cue_holds is not None else []
        suffix = ""
        if hold_ids:
            ids = ", ".join(f"#{hold_id}" for hold_id in hold_ids)
            suffix = f" (hold{'s' if len(hold_ids) > 1 else ''} {ids})"
        items.append((FEEDBACK[cue], cue + suffix))
    return items
