# image_quality.py
#
# Upload quality feedback for analog gauge photos. Nearly every CV/OCR
# failure investigated this session traced back to the same root cause:
# insufficient pixel information in small/close-up/blurry source photos —
# not a fixable algorithm choice (see gaugesdetectionplan.md / adaptive tick
# plan.md). That's outside this codebase's control per-photo, so the
# practical lever is surfacing an actionable warning to the caller so future
# uploads can be better, rather than silently returning a low-confidence
# analog reading with no explanation.
#
# Independent of gauge_reader.py's geometry logic on purpose — this is a
# generic "is this photo good enough" check, not gauge-specific math.

import cv2
import numpy as np

from gauge_reader import DialCircle

# A dial radius below this (in the original photo's own pixels) is where
# tick-mark detection kept failing this session (e.g. the close-up bar gauge
# fixture's ~92px-radius dial never yielded usable ticks, while larger dials
# fared better) — informal, empirically-set threshold, not a hard science.
_MIN_DIAL_RADIUS_PX = 120

# Laplacian variance is a standard sharpness metric — lower means blurrier.
_MIN_BLUR_SCORE = 60.0


def assess_blur(img_bgr: np.ndarray) -> float:
    """Laplacian variance of the image (or ROI) — higher is sharper."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def assess_gauge_quality(img_bgr: np.ndarray, circle: DialCircle | None) -> str | None:
    """Return a human-readable quality warning, or None if the photo looks
    adequate for analog gauge reading. Only meaningful when a dial circle was
    actually found — not applicable to plain text/digital images."""
    if circle is None:
        return None

    if circle.r < _MIN_DIAL_RADIUS_PX:
        return (
            f"The gauge appears small in this photo (~{int(circle.r * 2)}px across) — "
            "a closer photo may improve reading accuracy."
        )

    x0, y0 = max(int(circle.cx - circle.r), 0), max(int(circle.cy - circle.r), 0)
    x1, y1 = min(int(circle.cx + circle.r), img_bgr.shape[1]), min(int(circle.cy + circle.r), img_bgr.shape[0])
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blur_score = assess_blur(roi)
    if blur_score < _MIN_BLUR_SCORE:
        return "The gauge appears blurry in this photo — a sharper, steadier photo may improve reading accuracy."

    return None
