"""Draw hold, pose and feedback overlays on video frames."""

import math
import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .coaching import STATUS_COLORS, STATUS_NAMES, feedback_items
from .holds import HOLD_COLORS, Hold
from .pose import SKELETON


def draw_frame(frame: np.ndarray, holds: list[Hold], points: np.ndarray | None,
               valid: np.ndarray | None, metrics: list[str], cues: list[str],
               subtle_holds: bool = False, highlighted_holds: list[Hold] | None = None) -> None:
    height, width = frame.shape[:2]
    hold_layer = frame.copy() if subtle_holds else frame
    for hold in holds:
        x1, y1, x2, y2 = (int(round(v)) for v in hold.bbox)
        color = tuple(reversed(HOLD_COLORS[hold.category]))
        cv2.rectangle(hold_layer, (x1, y1), (x2, y2), color, 1)
        cv2.putText(hold_layer, f"{hold.id}:{hold.category}", (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    if subtle_holds:
        cv2.addWeighted(hold_layer, 0.45, frame, 0.55, 0, frame)
    if points is not None and valid is not None:
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                cv2.line(frame, tuple(np.rint(points[a]).astype(int)),
                         tuple(np.rint(points[b]).astype(int)), (0, 220, 255), 2)
        for point, visible in zip(points, valid):
            if visible:
                cv2.circle(frame, tuple(np.rint(point).astype(int)), 3, (0, 100, 255), -1)
    scale = max(0.35, min(0.65, width / 1600))
    line_height = max(15, int(25 * scale / 0.65))
    chars = max(12, int((width - 20) / (11 * scale)))
    lines = []
    for text in metrics + cues:
        lines.extend(textwrap.wrap(text, width=chars) or [""])
    lines = lines[:max(1, (height - 10) // line_height)]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, min(height, 8 + len(lines) * line_height)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (5, 5 + (index + 1) * line_height - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    for hold in highlighted_holds or []:
        x1, y1, x2, y2 = hold.bbox
        radius = max(8, math.ceil(math.hypot(x2 - x1, y2 - y1) / 2) + 4)
        cv2.circle(frame, tuple(round(v) for v in hold.center), radius, (0, 0, 255), 3)


def panel_height(width: int) -> int:
    # Even dimensions avoid silent frame truncation by common video codecs.
    return 2 * math.ceil(max(200, min(360, width * 0.65)) / 2)


def _font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default(size=size)


def _wrap_pixels(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=font) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def draw_feedback_frame(frame: np.ndarray, holds: list[Hold], points: np.ndarray | None,
                        valid: np.ndarray | None, metrics: list[str], cues: list[str],
                        highlighted_holds: list[Hold] | None = None) -> np.ndarray:
    """Append a readable feedback area below the video image."""
    height, width = frame.shape[:2]
    draw_frame(frame, holds, points, valid, metrics, [], subtle_holds=True,
               highlighted_holds=highlighted_holds)
    image = Image.new("RGB", (width, height + panel_height(width)), (13, 19, 30))
    image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
    draw = ImageDraw.Draw(image)
    items = feedback_items(cues)
    # Shrink text only as needed so all simultaneous cues remain visible.
    font_size = max(10, min(21, width // 30))
    while True:
        font = _font(font_size)
        label_font = _font(max(9, font_size - 2), bold=True)
        label_width = max(int(draw.textlength(name, font=label_font))
                          for name in STATUS_NAMES.values())
        text_width = max(20, width - label_width - 44)
        wrapped = [(kind, _wrap_pixels(draw, text, font, text_width)) for kind, text in items]
        line_height = font_size + 5
        card_heights = [max(29, len(lines) * line_height + 12) for _, lines in wrapped]
        if sum(card_heights) + 10 * len(items) + 14 <= panel_height(width) or font_size <= 9:
            break
        font_size -= 1
    y = height + 8
    for (kind, lines), card_height in zip(wrapped, card_heights):
        x1, x2 = 8, max(9, width - 8)
        draw.rounded_rectangle((x1, y, x2, y + card_height), radius=7,
                               fill=(24, 33, 47), outline=STATUS_COLORS[kind], width=2)
        draw.text((x1 + 8, y + 7), STATUS_NAMES[kind], font=label_font,
                  fill=STATUS_COLORS[kind])
        for index, line in enumerate(lines):
            draw.text((x1 + label_width + 16, y + 5 + index * line_height),
                      line, font=font, fill=(236, 242, 248))
        y += card_height + 8
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def pad_to_even(frame: np.ndarray) -> np.ndarray:
    """Pad odd dimensions so MP4 encoders do not silently crop the output."""
    height, width = frame.shape[:2]
    return cv2.copyMakeBorder(frame, 0, height % 2, 0, width % 2,
                              cv2.BORDER_CONSTANT, value=(30, 19, 13))
