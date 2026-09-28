# digital_display_reader.py
#
# Classical seven-segment LED/LCD display decoder, independent of
# gauge_reader.py's circle-based analog pipeline (a digital display isn't on
# a circular dial). See adaptive tick plan.md for why this exists: a VLM
# reading a photographed 7-segment display loses fine detail (particularly
# the decimal point) when the image is compressed/downscaled, but a segment
# is a binary lit/unlit signal and decimal points are small but fully-lit,
# high-contrast blobs — a much more tractable classical CV target than
# analog needle/tick reading. Approach follows the standard technique (see
# PyImageSearch "Recognizing digits with OpenCV and Python"; ssocr).
#
# Every stage can cleanly return None ("unknown") rather than guess, same
# discipline as gauge_reader.py.

import logging
import sys
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard 7-segment layout: a=top, b=top-right, c=bottom-right, d=bottom,
# e=bottom-left, f=top-left, g=middle. Segment order used everywhere below:
# (a, b, c, d, e, f, g).
_SEGMENT_PATTERNS = {
    (1, 1, 1, 1, 1, 1, 0): "0",
    (0, 1, 1, 0, 0, 0, 0): "1",
    (1, 1, 0, 1, 1, 0, 1): "2",
    (1, 1, 1, 1, 0, 0, 1): "3",
    (0, 1, 1, 0, 0, 1, 1): "4",
    (1, 0, 1, 1, 0, 1, 1): "5",
    (1, 0, 1, 1, 1, 1, 1): "6",
    (1, 1, 1, 0, 0, 0, 0): "7",
    (1, 1, 1, 1, 1, 1, 1): "8",
    (1, 1, 1, 1, 0, 1, 1): "9",
}

# Normalized (x, y) center of each segment's sampling patch within a digit's
# bounding box (0,0 = top-left, 1,1 = bottom-right), and the half-size of the
# patch to average over.
_SEGMENT_SAMPLE_POINTS = {
    "a": (0.50, 0.10), "b": (0.82, 0.30), "c": (0.82, 0.70),
    "d": (0.50, 0.90), "e": (0.18, 0.70), "f": (0.18, 0.30), "g": (0.50, 0.50),
}
_SEGMENT_ORDER = ("a", "b", "c", "d", "e", "f", "g")
_PATCH_HALF_W = 0.12
_PATCH_HALF_H = 0.08

_MIN_SAT = 70
_MIN_VAL = 70
_HUE_WINDOW = 15  # +/- degrees (OpenCV's 0-179 hue scale) treated as one color cluster
_MIN_CLUSTER_PIXELS = 200  # ignore tiny/noise hue clusters
_MAX_CANDIDATE_CLUSTERS = 5  # cap how many hue clusters get a decode attempt

_MIN_DIGIT_CELL_AREA_FRAC = 0.15  # relative to the largest cell found
_ON_FRACTION_THRESHOLD = 0.35  # fraction of a segment patch that must be lit


@dataclass
class DigitCell:
    x: int
    y: int
    w: int
    h: int


@dataclass
class DisplayReadResult:
    text: str
    cells: list


def _candidate_led_masks(img_bgr: np.ndarray):
    """Yield candidate "lit segment" masks, one per distinct saturated/bright
    hue present in the photo. Backlit digital displays vary in color (red/
    orange LED, blue or green LCD, etc.), so a fixed hue assumption doesn't
    generalize — and picking whichever hue has the most total pixels doesn't
    either, since a brightly colored housing/bezel can easily out-count a
    smaller display window (seen empirically: a yellow gauge housing
    outnumbering its own blue LCD). So instead of guessing which hue is "the
    display" up front, every distinct saturated hue cluster gets its own
    candidate mask, most-common first; the caller tries each in turn and
    naturally rejects non-display clusters when they fail to decode into
    valid 7-segment patterns — same decline-over-guess discipline as the
    rest of this module, just applied per-candidate.

    Within a candidate hue cluster, a hue threshold alone would find the
    whole display rectangle, not just the segments — the rest of the panel
    still shares the same hue at a similar saturation, just different
    brightness. So: Otsu-threshold brightness *within just that region* to
    split it into a bright group and a dim group — a fixed global brightness
    threshold can't do this since the two groups only separate cleanly
    relative to each other, not in absolute terms (validated empirically
    against real photos). Which group is actually "the segments" depends on
    display polarity and isn't knowable in advance: an emissive LED digit
    glows bright against a dark background, while a transmissive backlit LCD
    digit is a dark liquid-crystal shape blocking a bright backlight — the
    opposite arrangement. So both polarities are yielded as separate
    candidates; the caller's decode step naturally rejects whichever one
    doesn't actually form valid digit shapes."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    colorful = (s > _MIN_SAT) & (v > _MIN_VAL)
    if not colorful.any():
        return

    colorful_u8 = colorful.astype(np.uint8) * 255
    hue_hist = cv2.calcHist([h], [0], colorful_u8, [180], [0, 180]).flatten()
    h_int = h.astype(np.int16)

    claimed = np.zeros(180, dtype=bool)
    n_yielded = 0
    for hue in np.argsort(hue_hist)[::-1]:
        if n_yielded >= _MAX_CANDIDATE_CLUSTERS or hue_hist[hue] < _MIN_CLUSTER_PIXELS:
            break
        if claimed[hue]:
            continue  # already covered by a bigger, nearby cluster
        claimed[np.arange(hue - _HUE_WINDOW, hue + _HUE_WINDOW + 1) % 180] = True

        hue_dist = np.abs(h_int - int(hue))
        hue_dist = np.minimum(hue_dist, 180 - hue_dist)  # circular distance (hue wraps at 180)
        display_region = colorful & (hue_dist <= _HUE_WINDOW)
        if not display_region.any():
            continue

        thresh_val, _ = cv2.threshold(v[display_region], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bright_mask = np.zeros_like(v, dtype=np.uint8)
        bright_mask[display_region] = (v[display_region] > thresh_val).astype(np.uint8) * 255
        dark_mask = np.zeros_like(v, dtype=np.uint8)
        dark_mask[display_region] = (v[display_region] <= thresh_val).astype(np.uint8) * 255
        n_yielded += 1
        yield bright_mask
        yield dark_mask


def _deskew(img_bgr: np.ndarray, mask: np.ndarray) -> tuple:
    """Rotate the image+mask so the lit display region becomes horizontal —
    product photos often show the whole gauge (and its display) tilted."""
    points = cv2.findNonZero(mask)
    if points is None or len(points) < 10:
        return img_bgr, mask
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(points)
    if rw < rh:
        angle += 90
    if abs(angle) < 1.0:
        return img_bgr, mask
    rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    h, w = mask.shape[:2]
    rotated_img = cv2.warpAffine(img_bgr, rot, (w, h))
    rotated_mask = cv2.warpAffine(mask, rot, (w, h))
    return rotated_img, rotated_mask


_MAX_HEIGHT_RATIO = 1.3  # real digit cells in one row are all similarly tall
_MAX_CENTER_DEVIATION_FRAC = 0.3  # vertical-center wobble allowed, relative to height


def _find_digit_cells(mask: np.ndarray) -> list:
    """A single digit's lit segments touch each other at shared corners
    (e.g. top bar meets top-right bar), so they already form one connected
    component per digit with no merging needed — validated empirically
    against a real photo. The decimal point is a separate, much smaller
    component, filtered out here by relative size (real digits are all
    similarly sized; the dot is a small fraction of that).

    Since _candidate_led_masks now tries several color/polarity guesses per
    photo, a mask built from the wrong region (background clutter, a logo)
    can still leave behind components that individually look plausible —
    but real digits sit in one row at one consistent height, while
    incidental clutter doesn't. Any component that doesn't match the
    others' height and vertical position is a sign this candidate mask
    isn't a real digit row at all, so the whole candidate is discarded
    rather than trying to salvage a subset of it (seen empirically: a
    non-display hue cluster produced scattered components at wildly
    different heights/positions that nonetheless decoded into a
    confident-looking but wrong value)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return []

    boxes = [tuple(stats[i, :5]) for i in range(1, n)]
    max_area = max(b[4] for b in boxes)
    boxes = [b for b in boxes if b[4] >= max_area * _MIN_DIGIT_CELL_AREA_FRAC]
    if len(boxes) < 2:
        return []

    heights = sorted(b[3] for b in boxes)
    median_h = heights[len(heights) // 2]
    centers = [b[1] + b[3] / 2 for b in boxes]
    median_cy = sorted(centers)[len(centers) // 2]

    for b, cy in zip(boxes, centers):
        if not (median_h / _MAX_HEIGHT_RATIO <= b[3] <= median_h * _MAX_HEIGHT_RATIO):
            return []
        if abs(cy - median_cy) > median_h * _MAX_CENTER_DEVIATION_FRAC:
            return []

    boxes.sort(key=lambda b: b[0])
    return [DigitCell(x=b[0], y=b[1], w=b[2], h=b[3]) for b in boxes]


def _sample_segment(mask: np.ndarray, cell: DigitCell, name: str) -> bool:
    cx, cy = _SEGMENT_SAMPLE_POINTS[name]
    px = cell.x + cx * cell.w
    py = cell.y + cy * cell.h
    hw, hh = _PATCH_HALF_W * cell.w, _PATCH_HALF_H * cell.h
    x0, y0 = max(int(px - hw), 0), max(int(py - hh), 0)
    x1, y1 = min(int(px + hw), mask.shape[1]), min(int(py + hh), mask.shape[0])
    if x1 <= x0 or y1 <= y0:
        return False
    patch = mask[y0:y1, x0:x1]
    return (patch > 0).mean() >= _ON_FRACTION_THRESHOLD


def _decode_digit(mask: np.ndarray, cell: DigitCell) -> str:
    pattern = tuple(1 if _sample_segment(mask, cell, s) else 0 for s in _SEGMENT_ORDER)
    return _SEGMENT_PATTERNS.get(pattern)


_DECIMAL_MAX_SIZE_FRAC = 0.35  # a decimal point blob is much smaller than a digit cell
_DECIMAL_SEARCH_GAP_FRAC = 1.5  # search this many cell-widths to the right of a digit for its decimal point


def _find_decimal_after(mask: np.ndarray, cell: DigitCell, next_cell_x: int) -> bool:
    """A decimal point sits low and small just after a digit cell, distinct
    from the next digit's own segments (checked by requiring it be small and
    isolated, and before the next digit cell actually starts)."""
    search_x0 = cell.x + cell.w
    search_x1 = min(cell.x + int(cell.w * (1 + _DECIMAL_SEARCH_GAP_FRAC)), next_cell_x)
    if search_x1 <= search_x0:
        return False
    y0 = cell.y + int(cell.h * 0.65)
    y1 = cell.y + cell.h
    region = mask[max(y0, 0):min(y1, mask.shape[0]), max(search_x0, 0):min(search_x1, mask.shape[1])]
    if region.size == 0:
        return False
    n, _, stats, _ = cv2.connectedComponentsWithStats(region)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if area > 0 and w < cell.w * _DECIMAL_MAX_SIZE_FRAC and h < cell.h * _DECIMAL_MAX_SIZE_FRAC:
            return True
    return False


def _decode_cells(mask: np.ndarray, cells: list) -> DisplayReadResult | None:
    digits = []
    for cell in cells:
        d = _decode_digit(mask, cell)
        if d is None:
            return None  # ambiguous segment pattern — decline rather than guess
        digits.append(d)

    text = ""
    for i, (cell, d) in enumerate(zip(cells, digits)):
        text += d
        next_x = cells[i + 1].x if i + 1 < len(cells) else mask.shape[1]
        if _find_decimal_after(mask, cell, next_x):
            text += "."

    return DisplayReadResult(text=text, cells=cells)


_MIN_DIGIT_ASPECT = 0.35  # width/height lower bound for one plausible digit cell
_MAX_DIGIT_ASPECT = 0.70  # width/height upper bound for one plausible digit cell
_MAX_GRID_DIGITS = 8


def _largest_component_bbox(mask: np.ndarray):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    idx = int(np.argmax(stats[1:, 4])) + 1
    x, y, w, h = stats[idx, :4]
    return (int(x), int(y), int(w), int(h))


def _decode_merged_blob(mask: np.ndarray) -> DisplayReadResult | None:
    """Fallback for when segments don't separate into distinct per-digit
    connected components at all (e.g. thin, blurred strokes bridging
    adjacent digits at low photo resolution — one wide blob instead of N
    digit-sized ones). Rather than requiring real component boundaries,
    guess the digit count from the merged blob's aspect ratio against a
    typical single 7-segment digit's aspect ratio, then decode fixed-width
    grid slices directly. Still declines rather than guesses: only accepts
    a digit count if it's the ONE candidate count (out of every plausible
    count for this blob's width) whose slices all decode to unambiguous
    digits — a wrong slice count usually leaves at least one slot's sampled
    pattern not matching any real digit."""
    bbox = _largest_component_bbox(mask)
    if bbox is None:
        return None
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return None

    lo = int(w / (h * _MAX_DIGIT_ASPECT))
    hi = min(_MAX_GRID_DIGITS, int(w / (h * _MIN_DIGIT_ASPECT)) + 1)
    if lo < 2 or hi < lo:
        return None  # blob's own aspect ratio doesn't already imply multiple digits

    results = {}
    for n in range(lo, hi + 1):
        slot_w = w / n
        cells = [DigitCell(x=int(round(x + i * slot_w)), y=y, w=int(round(slot_w)), h=h) for i in range(n)]
        result = _decode_cells(mask, cells)
        if result is not None:
            results[n] = result

    if len(results) != 1:
        return None  # no clean count, or more than one count decoded cleanly (ambiguous)
    return next(iter(results.values()))


def _decode_from_mask(img_bgr: np.ndarray, mask: np.ndarray) -> DisplayReadResult | None:
    img_bgr, mask = _deskew(img_bgr, mask)
    cells = _find_digit_cells(mask)
    if len(cells) >= 2:
        return _decode_cells(mask, cells)
    return _decode_merged_blob(mask)


def read_digital_display(img_bgr: np.ndarray) -> DisplayReadResult | None:
    """Attempt to decode a 7-segment LED/LCD display. Tries every candidate
    display-color/polarity combination (see _candidate_led_masks) rather
    than stopping at the first one that decodes cleanly — a candidate mask
    built from the wrong hue cluster (e.g. bezel text, a logo) can still
    coincidentally form shapes that decode into *some* valid-looking digit
    pattern, so "decodes without error" alone isn't proof it's the real
    display (seen empirically: an unrelated cluster decoded to a
    confident-looking but wrong value). Only trust the result when every
    candidate that successfully decodes agrees on the same text — if none
    decode, or if they disagree, decline rather than guess which one is
    real."""
    try:
        results = []
        for mask in _candidate_led_masks(img_bgr):
            result = _decode_from_mask(img_bgr, mask)
            if result is not None:
                results.append(result)

        if not results:
            return None
        distinct_texts = {r.text for r in results}
        if len(distinct_texts) > 1:
            logger.warning(
                "Candidate display regions disagree on decoded value (%s) — declining",
                distinct_texts,
            )
            return None
        return results[0]
    except Exception as e:
        logger.warning("Digital display decode failed: %s", e)
        return None


def _debug_visualize(image_path: str) -> None:
    """Dev-only tool: draws the LED mask and detected digit cells for each
    candidate hue cluster tried, prints intermediate values. Run as:
    python digital_display_reader.py <image_path>"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    for i, mask in enumerate(_candidate_led_masks(img)):
        cv2.imwrite(f"digital_debug_mask_raw_{i}.png", mask)

        rotated_img, rotated_mask = _deskew(img, mask)
        cv2.imwrite(f"digital_debug_mask_deskewed_{i}.png", rotated_mask)
        cv2.imwrite(f"digital_debug_deskewed_{i}.png", rotated_img)

        cells = _find_digit_cells(rotated_mask)
        print(f"[candidate {i}] digit cells ({len(cells)}): {cells}")

        overlay = rotated_img.copy()
        digits = []
        for cell in cells:
            d = _decode_digit(rotated_mask, cell)
            digits.append(d)
            color = (0, 255, 0) if d is not None else (0, 0, 255)
            cv2.rectangle(overlay, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), color, 2)
            cv2.putText(overlay, str(d), (cell.x, cell.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        print(f"[candidate {i}] decoded digits: {digits}")
        cv2.imwrite(f"digital_debug_overlay_{i}.png", overlay)

    result = read_digital_display(img)
    print(f"read_digital_display result: {result}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python digital_display_reader.py <image_path>")
        sys.exit(1)
    _debug_visualize(sys.argv[1])
