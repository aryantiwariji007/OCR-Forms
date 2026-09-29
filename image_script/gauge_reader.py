# gauge_reader.py
#
# Deterministic analog-gauge needle reading via classical CV, used to replace
# the VLM's unreliable visual needle-angle interpolation (see vlm_client.py /
# gaugesdetectionplan.md for why). Pure OpenCV + math, no FastAPI dependency —
# independently testable.
#
# Angle convention used throughout this module: angle = atan2(y - cy, x - cx) % 360
# in raw image pixel coordinates (y grows downward). That gives 0 deg at 3
# o'clock, 90 deg at 6 o'clock, 180 deg at 9 o'clock, 270 deg at 12 o'clock —
# i.e. clockwise-increasing. This matches the clockwise-increasing-value
# assumption used for scale calibration below. Getting this sign wrong
# silently produces a backwards-but-plausible-looking reading, so every
# angle in this module must use this exact formula.
#
# Pipeline: find the dial (Hough circle) -> find the needle (Hough lines on
# an ink mask, filtered to segments with one endpoint near the center) ->
# find ordered tick-mark angles (Hough lines on the same ink mask, filtered
# to a band near the rim) -> assign values to those ticks by linearly
# interpolating between externally-provided min/max scale values -> interpolate
# the needle's angle between the two nearest calibrated ticks.
#
# Why min/max come from outside this module: generic OCR (RapidOCR, and
# independently a trained scene-text pipeline from ethz-asl/analog_gauge_reader)
# cannot read the tiny printed scale digits on real close-up gauge photos, at
# any preprocessing tried (see gaugesdetectionplan.md) — the digits are only
# ~5-10px wide at native resolution, a genuine information floor, not an
# algorithm problem. A VLM reading the two labeled endpoint values as plain
# text is reliable (it's needle-angle *interpolation* VLMs are bad at, not
# reading legible printed numbers), so callers get min/max from the VLM and
# pass them in here. Every stage can cleanly return None ("unknown") rather
# than guess.

import argparse
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DialCircle:
    cx: float
    cy: float
    r: float


@dataclass
class CalibrationTick:
    value: float
    angle_deg: float


@dataclass
class GaugeReadResult:
    value: float
    circle: DialCircle
    needle_angle: float
    ticks_used: list


def _angle_deg(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dy, dx)) % 360


def angle_to_clock_position(angle_deg: float) -> str:
    """Convert this module's angle convention (0 deg = 3 o'clock, clockwise-
    increasing — see module docstring) into an intuitive clock-face
    description, e.g. "around 1 o'clock" or "between 7 and 8 o'clock" — for
    describing a CV-measured angle to a VLM in the same terms a human
    looking at the photo would use (see the tool-augmented-VLM approach in
    adaptive tick plan.md)."""
    hour_float = ((angle_deg + 90) / 30) % 12  # in [0, 12)
    lower = int(math.floor(hour_float)) % 12
    frac = hour_float - math.floor(hour_float)
    lower_display = 12 if lower == 0 else lower

    if frac < 0.15 or frac > 0.85:
        nearest = round(hour_float) % 12
        nearest_display = 12 if nearest == 0 else nearest
        return f"around {nearest_display} o'clock"

    upper = (lower + 1) % 12
    upper_display = 12 if upper == 0 else upper
    return f"between {lower_display} and {upper_display} o'clock"


def clock_position_to_angle(hour: float) -> float:
    """Inverse of angle_to_clock_position: convert a clock hour (1-12, 12 or
    0 both meaning 12 o'clock) back into this module's angle convention."""
    return (hour * 30 - 90) % 360


def interpolate_from_two_anchors(needle_angle: float, val1: float, hour1: float, val2: float, hour2: float) -> float | None:
    """Deterministically interpolate needle_angle's value between two
    (value, clock-hour) anchors — used when a VLM reports the clock
    positions of the scale's min/max labels (a plain reading task, already
    reliably done for their VALUES via the "[scale:...]" tag) instead of
    being asked to judge the needle's position itself (unreliable — see
    adaptive tick plan.md: VLMs complied with an injected needle-position
    hint only ~1 in 6 times when still asked to render the final judgment).
    Removing that judgment call entirely makes the measurement authoritative
    every time instead of probabilistically.

    Deliberately uses the two FAR-APART scale endpoints (not the two ticks
    immediately bracketing the needle) as anchors: adjacent close-together
    ticks are easy for a VLM to misjudge by one hour in a way that flips
    their apparent clockwise order and corrupts the whole calculation — far
    apart, unambiguous endpoints don't have that failure mode. This always
    assumes the clockwise-increasing convention documented at the top of
    this module (true for the vast majority of gauges), taking the full
    clockwise sweep from hour1 to hour2 rather than the shorter arc — unlike
    two adjacent ticks, real min/max endpoints legitimately span most of the
    dial (~270 deg), so "shorter arc" would pick the wrong direction here.
    Returns None if the anchors are degenerate, OR if the needle falls
    meaningfully outside the stated endpoint range — clamping that case to
    exactly the boundary value instead would silently paper over the
    endpoints' clock positions being wrong (this model's clock-position
    *judgment* itself has real error, independent of the order-flip issue
    the far-apart-anchor design already fixes — see adaptive tick plan.md),
    turning a bad reading into a deterministically WRONG one every time
    rather than an honest decline."""
    if val1 == val2:
        return None
    if val1 > val2:
        val1, hour1, val2, hour2 = val2, hour2, val1, hour1

    a1, a2 = clock_position_to_angle(hour1), clock_position_to_angle(hour2)
    a2 = a1 + ((a2 - a1) % 360)
    if a2 - a1 <= 0:
        return None

    na = min((needle_angle + k * 360 for k in (-1, 0, 1)), key=lambda x: min(abs(x - a1), abs(x - a2)))
    span = a2 - a1
    frac = (na - a1) / span
    tolerance = 0.05
    if frac < -tolerance or frac > 1 + tolerance:
        return None
    frac = max(0.0, min(1.0, frac))
    return val1 + frac * (val2 - val1)


def _resize_for_detection(img: np.ndarray, target_max_dim: int = 1000) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = target_max_dim / max(h, w)
    if scale >= 1.0:
        return img, 1.0
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


# A candidate circle's fraction of its own radius that falls outside the photo
# frame (0 = fully inside, 1 = a full radius' worth of overflow). A real dial
# is normally framed whole in a close-up photo, so a spurious Hough circle
# (an artifact of a tag plate, rivets, or bezel texture forming a rough ring)
# is much more likely to be one that's mostly clipped by the frame edge —
# measured across every fixture: every correctly-picked circle overflows
# 0-9%, while wrong picks measured 44-130%.
_MAX_CIRCLE_OVERFLOW_FRAC = 0.3


def _circle_overflow_frac(cx: int, cy: int, r: int, w: int, h: int) -> float:
    if r <= 0:
        return float("inf")
    overflow = max(0, r - cx) + max(0, (cx + r) - w) + max(0, r - cy) + max(0, (cy + r) - h)
    return overflow / r


def find_dial_circle(img_bgr: np.ndarray) -> DialCircle | None:
    """Locate the gauge's circular face via Hough Circle Transform."""
    small, scale = _resize_for_detection(img_bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    min_dim = min(h, w)

    for param2, min_r_frac, max_r_frac in ((60, 0.15, 0.48), (40, 0.08, 0.49)):
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.5,
            minDist=min_dim * 0.5,
            param1=100, param2=param2,
            minRadius=int(min_dim * min_r_frac),
            maxRadius=int(min_dim * max_r_frac),
        )
        if circles is None:
            continue
        rounded = np.round(circles[0]).astype(int)
        # Among Hough's candidates, the largest radius isn't necessarily the
        # real dial — a spurious circle elsewhere in the photo can come out
        # larger by a narrow, essentially arbitrary margin (seen on a real
        # photo: a false circle near a tag plate won by 12px while extending
        # almost an entire radius past the bottom of the frame). Prefer
        # candidates that actually fit within the photo first, then take the
        # largest among those; fall back to the full candidate set only if
        # none fit, so a genuinely tightly-cropped real dial still gets a
        # result instead of nothing.
        in_frame = [c for c in rounded if _circle_overflow_frac(*c, w, h) <= _MAX_CIRCLE_OVERFLOW_FRAC]
        pool = in_frame if in_frame else rounded
        cx, cy, r = max(pool, key=lambda c: c[2])
        return DialCircle(cx / scale, cy / scale, r / scale)
    return None


_INK_SAT_THRESH = 60
_INK_VAL_THRESH = 140


def _ink_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary mask of 'ink' pixels: low-saturation (excludes colored scale
    arcs/bands) and dark (excludes the white/light face) — isolates the
    needle, tick marks, and printed text from the rest of the dial. A plain
    grayscale threshold merges the colored arc into the same blob as the
    real ink (the arc is dark-ish in grayscale too); using saturation as a
    second axis is what actually separates them — validated empirically
    against real gauge photos (see gaugesdetectionplan.md).

    Uses a fixed brightness cutoff, so it only holds for a light-faced dial.
    Prefer _face_ink_mask whenever the dial circle is known."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return ((s < _INK_SAT_THRESH) & (v < _INK_VAL_THRESH)).astype(np.uint8) * 255


def _face_ink_mask(bgr: np.ndarray, lcx: float, lcy: float, r: float) -> tuple:
    """Ink mask for a dial whose circle is already known, thresholded
    *relative to that dial's own face* rather than against a fixed cutoff.

    _ink_mask's fixed "darker than 140" rule silently assumes a light face.
    On a dim photo of a dark or greyish face the entire face reads as ink —
    measured 42.6% of the face on one real industrial gauge, against 9-28%
    on photos that work — and Hough then returns a confident-looking angle
    computed from what is effectively noise. Otsu within the face splits
    whatever is actually there into its own dark and light groups, which is
    the same relative-threshold fix that made digital_display_reader._led_mask
    work across displays of different colors. The low-saturation constraint is
    kept as-is; that axis is what separates ink from colored arcs and is
    unaffected by overall brightness.

    Returns (ink, face_mask) so callers can measure coverage against the face
    rather than the whole crop."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    face_mask = np.zeros(v.shape, np.uint8)
    cv2.circle(face_mask, (int(lcx), int(lcy)), max(int(r * 0.95), 1), 255, -1)
    inside = face_mask > 0
    if not inside.any():
        return np.zeros(v.shape, np.uint8), face_mask

    threshold, _ = cv2.threshold(v[inside], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = np.zeros(v.shape, np.uint8)
    ink[inside] = ((v[inside] < threshold) & (s[inside] < _INK_SAT_THRESH)).astype(np.uint8) * 255
    return ink, face_mask


def _closed_for_lines(ink: np.ndarray) -> np.ndarray:
    """A glossy/metal needle can carry a bright specular highlight down its own
    centre — bright enough to cross the Otsu threshold and read as "not ink",
    so the needle shows up as a thin hollow outline (just its two edges)
    instead of one solid shape. Hough then can't trace a single line through
    it at all and returns noise from elsewhere on the dial (seen on a real
    photo: top candidates landed on bezel rust and label text, nowhere near
    the actual needle). A small morphological close bridges that gap without
    visibly changing shapes that were already solid.

    Deliberately NOT folded into _face_ink_mask itself: closing can merge the
    pivot hub with a sliver of adjacent ink unevenly, shifting its measured
    centroid enough to look like a bad circle fit (seen on a real photo: hub
    offset went from 0.08r to 0.17r, above the decline threshold, for a needle
    that had nothing wrong with it). _hub_offset must keep using the raw mask;
    only line-finding and thickness measurement need the closed one."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel)


_ENHANCE_UPSCALE = 2.0


def _enhance_roi(bgr: np.ndarray, lcx: float, lcy: float, r: float) -> tuple:
    """Contrast-boost (CLAHE on the HSV value channel) and upscale a dial
    crop before ink-mask thresholding. Unlike OCR — a hard information floor
    no enhancement fixed, tested extensively (see gaugesdetectionplan.md) —
    Hough-based edge/line detection depends on edge crispness rather than
    resolving legible letterforms, so it may respond differently. Returns
    (enhanced_bgr, lcx, lcy, r) all rescaled consistently, since angles
    measured from a fixed center are invariant to this uniform scaling."""
    up = _ENHANCE_UPSCALE
    resized = cv2.resize(bgr, (int(bgr.shape[1] * up), int(bgr.shape[0] * up)), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_enhanced = clahe.apply(v)
    enhanced = cv2.cvtColor(cv2.merge([h, s, v_enhanced]), cv2.COLOR_HSV2BGR)
    return enhanced, lcx * up, lcy * up, r * up


# Even with a face-relative mask, a wildly over-filled result means the dial
# isn't separable into markings and background at all. Loose sanity net only —
# the adaptive mask brings real photos to 10-25% of the face.
_MAX_INK_COVERAGE_FRAC = 0.5

# Radii (as fractions of dial radius) at which needle thickness is sampled, and
# compared side-by-side at each one. Deliberately close to the hub and stopping
# well short of the rim: a counterweight is short, so sampling too far out
# measures past its end into background (reading as falsely "thin"); some
# pointers also flare into an arrowhead near their own far tip, which measuring
# too far out picks up and reads as falsely "thick". Measured across 4 real
# needle shapes (simple wedge counterweights and a diamond-tail/arrowhead-tip
# needle): the counterweight side was wider at every one of these radii except
# one single-point flip on one photo, while going past 0.30r started flipping
# on the arrowhead needle specifically. A per-radius vote (below) absorbs that
# kind of single-point noise; a plain average across a range that runs too far
# out cannot, because it has no way to discard the contaminated samples.
_THICKNESS_SAMPLE_RADII = (0.10, 0.14, 0.18, 0.22, 0.26)
_THICKNESS_HALF_SPAN = 0.18  # how far to sweep perpendicular to the needle, as a fraction of r

# Fraction of sampled radii that must agree on which side is thicker before
# that side is trusted as the counterweight. Set well above a simple majority
# so a couple of noisy samples can't flip the outcome.
_NEEDLE_DIRECTION_VOTE_FRAC = 0.7

# Below this (fraction of r), nothing needle-like is present at a given radius
# — e.g. past the physical end of a short counterweight, or a digital gauge's
# round housing with no pointer at all. A radius this thin on either side isn't
# evidence of anything and is left out of the vote rather than counted as "the
# thin side".
_MIN_NEEDLE_WIDTH_FRAC = 0.02

# Every angle here is measured from the detected circle's centre, so that centre
# has to actually be the needle's pivot. HoughCircles can settle on a bezel or
# an off-centre fit — on a real photo it landed 0.21r away from the true hub,
# which skewed the reading by ~30 degrees while still looking plausible. Good
# fits measured 0.02-0.08r, so this threshold separates them with margin.
_MAX_HUB_OFFSET_FRAC = 0.12


def _hub_offset(ink: np.ndarray, lcx: float, lcy: float, r: float) -> float | None:
    """Distance from the detected dial centre to the nearest substantial ink
    blob that could be the needle's pivot hub, as a fraction of r. None when no
    such blob exists near the centre at all."""
    count, _, stats, centroids = cv2.connectedComponentsWithStats(ink)
    min_area = (r * 0.04) ** 2
    nearest = None
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        distance = math.hypot(centroids[i][0] - lcx, centroids[i][1] - lcy)
        if distance < r * 0.45 and (nearest is None or distance < nearest):
            nearest = distance
    return nearest / r if (nearest is not None and r > 0) else None


def _width_at_radius(ink: np.ndarray, lcx: float, lcy: float, r: float, angle_deg: float, frac: float) -> float:
    """Ink width perpendicular to a ray, at one specific radius fraction, as a
    fraction of the dial radius."""
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    px, py = -dy, dx  # unit vector perpendicular to the ray
    h, w = ink.shape[:2]
    step = max(r * 0.005, 0.5)
    span = _THICKNESS_HALF_SPAN * r

    bx, by = lcx + frac * r * dx, lcy + frac * r * dy
    lit = 0
    offset = -span
    while offset < span:
        x, y = int(round(bx + offset * px)), int(round(by + offset * py))
        if 0 <= x < w and 0 <= y < h and ink[y, x] > 0:
            lit += 1
        offset += step
    return lit * step / r if r > 0 else 0.0


def _pick_needle_direction(ink: np.ndarray, lcx: float, lcy: float, r: float, angle_deg: float) -> float | None:
    """Resolve which end of the needle's line is the pointing tip, or None if
    it can't be told confidently.

    Hough finds the needle's line but not its direction, and many gauges carry
    a short, blunt counterweight opposite the pointer — reading that end puts
    the answer ~180 degrees out (seen on a real photo: a counterweight aimed at
    "4" was read as 3.7 bar when the tip sat just above 0). The discriminator is
    thickness rather than length: a pointer tapers thin while a counterweight is
    stubby and wide, and thickness can be measured close to the hub where both
    ends definitely exist. Length is the more obvious signal but a worse one in
    practice — measuring outward runs into tick marks and printed numbers that
    bridge onto the counterweight and reverse the comparison (tried, and it
    flipped three otherwise-correct photos).

    Compares thickness at each radius in _THICKNESS_SAMPLE_RADII separately and
    takes a vote, rather than averaging them into one number — a single fixed
    "the widest point" radius doesn't generalize across needle shapes (an
    arrowhead-tipped needle's flare and a diamond-shaped counterweight's bulge
    can sit at different distances from the hub), and averaging lets one
    contaminated sample (e.g. a radius that's already past a short
    counterweight's end) quietly cancel out several good ones. A vote can
    simply discard that one bad sample instead.

    Declines rather than guessing when the vote is too split to call, since a
    coin flip here produces a confidently wrong reading roughly half the time."""
    forward_thinner_votes = 0
    reverse_thinner_votes = 0
    counted = 0
    max_width_seen = 0.0

    for frac in _THICKNESS_SAMPLE_RADII:
        forward = _width_at_radius(ink, lcx, lcy, r, angle_deg, frac)
        reverse = _width_at_radius(ink, lcx, lcy, r, angle_deg + 180.0, frac)
        max_width_seen = max(max_width_seen, forward, reverse)
        if forward < _MIN_NEEDLE_WIDTH_FRAC and reverse < _MIN_NEEDLE_WIDTH_FRAC:
            continue  # off the ink on both sides at this radius -- not informative
        counted += 1
        if forward < reverse:
            forward_thinner_votes += 1
        elif reverse < forward:
            reverse_thinner_votes += 1

    if max_width_seen < _MIN_NEEDLE_WIDTH_FRAC:
        logger.info("No needle-like ink near the hub (max width %.3fr) — declining", max_width_seen)
        return None
    if counted == 0:
        logger.info("No informative radii to judge needle direction — declining")
        return None

    # The tapered (thinner) end is the pointer.
    if forward_thinner_votes >= counted * _NEEDLE_DIRECTION_VOTE_FRAC:
        return angle_deg
    if reverse_thinner_votes >= counted * _NEEDLE_DIRECTION_VOTE_FRAC:
        return (angle_deg + 180.0) % 360.0

    logger.info(
        "Needle ends too similar to tell tip from counterweight (forward thinner "
        "at %d/%d radii, reverse thinner at %d/%d) — declining",
        forward_thinner_votes, counted, reverse_thinner_votes, counted,
    )
    return None


def find_needle_angle(img_bgr: np.ndarray, circle: DialCircle) -> float | None:
    """Locate the needle within the dial and return its angle from center.

    The needle is distinguished from tick marks by having one endpoint very
    close to the dial's center (the pivot) while tick marks live near the rim
    and never pass near center. Which end of that line is the pointing tip is
    resolved separately — see _pick_needle_direction."""
    small, scale = _resize_for_detection(img_bgr)
    cx, cy, r = circle.cx * scale, circle.cy * scale, circle.r * scale
    h, w = small.shape[:2]
    x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
    x1, y1 = min(int(cx + r), w), min(int(cy + r), h)
    roi = small[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    lcx, lcy = cx - x0, cy - y0
    roi, lcx, lcy, r = _enhance_roi(roi, lcx, lcy, r)

    ink, face_mask = _face_ink_mask(roi, lcx, lcy, r)

    face_px = int(np.count_nonzero(face_mask))
    if face_px == 0:
        return None
    ink_coverage = np.count_nonzero(ink) / face_px
    if ink_coverage > _MAX_INK_COVERAGE_FRAC:
        logger.warning(
            "Ink mask covers %.0f%% of the dial face — not separable into markings "
            "and background; declining",
            ink_coverage * 100,
        )
        return None

    # Hub-fit validity is checked against the RAW mask — closing (below) can
    # merge the hub with a sliver of adjacent ink unevenly and shift its
    # centroid, which looks exactly like a bad circle fit even when the fit is
    # fine (see _closed_for_lines).
    hub_offset = _hub_offset(ink, lcx, lcy, r)
    if hub_offset is None or hub_offset > _MAX_HUB_OFFSET_FRAC:
        logger.warning(
            "Detected dial centre doesn't line up with a pivot hub (offset %s) — the "
            "circle fit is unreliable, so any angle from it would be too; declining",
            "none found" if hub_offset is None else f"{hub_offset:.2f}r",
        )
        return None

    lines_ink = _closed_for_lines(ink)

    # The needle's near-center endpoint lands at the edge of the pivot hub, not
    # the exact geometric center pixel — measured ~0.2-0.35r away in practice —
    # so center_tol must be looser than a naive "near center" guess.
    center_tol = r * 0.35
    min_far = r * 0.4
    min_line_length = max(int(r * 0.3), 1)
    max_line_gap = max(int(r * 0.05), 1)

    lines = cv2.HoughLinesP(
        lines_ink, 1, np.pi / 180, threshold=30,
        minLineLength=min_line_length, maxLineGap=max_line_gap,
    )
    if lines is None:
        return None

    candidates = []
    for x1_, y1_, x2_, y2_ in lines[:, 0]:
        d1 = math.hypot(x1_ - lcx, y1_ - lcy)
        d2 = math.hypot(x2_ - lcx, y2_ - lcy)
        if d1 <= center_tol and d2 >= min_far:
            far_x, far_y, length = x2_, y2_, d2
        elif d2 <= center_tol and d1 >= min_far:
            far_x, far_y, length = x1_, y1_, d1
        else:
            continue
        angle = _angle_deg(far_x - lcx, far_y - lcy)
        candidates.append((length, angle))

    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    return _pick_needle_direction(lines_ink, lcx, lcy, r, candidates[0][1])


# Tuned for tick MARKS (short radial dashes near the rim), not tick TEXT — see
# module docstring for why marks are detected geometrically instead of
# reading digit positions via OCR.
_TICK_LENGTH_MIN_FRAC = 0.03
_TICK_LENGTH_MAX_FRAC = 0.22
_TICK_RADIAL_TOL_DEG = 12.0  # max angle between a segment's own orientation and the radial direction at its midpoint
_TICK_CLUSTER_TOL_DEG = 4.0  # merge Hough segments this close into one tick

# Adaptive band search: a fixed fractional band (e.g. "0.55-0.92 of r") was
# tried and found not to generalize — it missed real ticks on one gauge photo
# and locked onto the outer bezel edge on another, differently-proportioned
# one (see adaptive tick plan.md). Instead, slide a fixed-width band across
# candidate radii and score each by how many *self-consistent* (evenly
# angularly spaced) ticks it finds — real tick marks are evenly spaced,
# bezel/text/arc artifacts generally aren't.
_BAND_WIDTH_FRAC = 0.28
_BAND_MIN_OUTER = 0.35
_BAND_MAX_OUTER = 1.0
_BAND_STEP = 0.05
_MIN_TICKS_FOR_BAND = 3
_MAX_GAP_COEFF_VAR = 0.6  # max allowed (stdev/mean) of angular gaps between ticks


def _detect_ticks_in_band(ink: np.ndarray, lcx: float, lcy: float, r: float,
                           band_inner: float, band_outer: float) -> list:
    """Run the Hough-line + radial-orientation tick detector restricted to a
    single candidate band [band_inner*r, band_outer*r] from center."""
    band_mask = np.zeros_like(ink)
    cv2.circle(band_mask, (int(lcx), int(lcy)), max(int(r * band_outer), 1), 255, -1)
    cv2.circle(band_mask, (int(lcx), int(lcy)), max(int(r * band_inner), 0), 0, -1)
    banded = cv2.bitwise_and(ink, band_mask)

    min_len = max(int(r * _TICK_LENGTH_MIN_FRAC), 3)
    max_len = max(int(r * _TICK_LENGTH_MAX_FRAC), min_len + 1)
    lines = cv2.HoughLinesP(banded, 1, np.pi / 180, threshold=15,
                             minLineLength=min_len, maxLineGap=2)
    if lines is None:
        return []

    angles = []
    for lx1, ly1, lx2, ly2 in lines[:, 0]:
        length = math.hypot(lx2 - lx1, ly2 - ly1)
        if not (min_len <= length <= max_len):
            continue
        mx, my = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        radial_angle = _angle_deg(mx - lcx, my - lcy)
        seg_angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1)) % 180
        radial_dir = radial_angle % 180
        diff = abs(seg_angle - radial_dir)
        diff = min(diff, 180 - diff)
        if diff > _TICK_RADIAL_TOL_DEG:
            continue
        angles.append(radial_angle)

    return _cluster_angles(angles, _TICK_CLUSTER_TOL_DEG)


def _band_score(angles: list) -> tuple | None:
    """(count, gap_coeff_var) if angles look like real, evenly-spaced ticks,
    else None. Lower gap_coeff_var = more evenly spaced = more likely real."""
    if len(angles) < _MIN_TICKS_FOR_BAND:
        return None
    ordered = _order_ticks_by_gap(angles)
    gaps = [ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1)]
    if not gaps:
        return None
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return None
    variance = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    coeff_var = math.sqrt(variance) / mean_gap
    return (len(angles), coeff_var)


def _find_tick_mark_angles(img_bgr: np.ndarray, circle: DialCircle) -> list:
    """Detect scale tick marks via Hough lines on the ink mask, searching
    across candidate radius bands and picking the one whose ticks are most
    self-consistent (see _band_score) — distinguishes real, evenly-spaced
    tick marks from bezel edges or stray text at a fixed radius fraction."""
    small, scale = _resize_for_detection(img_bgr)
    cx, cy, r = circle.cx * scale, circle.cy * scale, circle.r * scale
    h, w = small.shape[:2]
    x0, y0 = max(int(cx - r), 0), max(int(cy - r), 0)
    x1, y1 = min(int(cx + r), w), min(int(cy + r), h)
    roi = small[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    lcx, lcy = cx - x0, cy - y0
    roi, lcx, lcy, r = _enhance_roi(roi, lcx, lcy, r)
    ink, _ = _face_ink_mask(roi, lcx, lcy, r)

    best_key = None
    best_angles = []
    outer = _BAND_MIN_OUTER + _BAND_WIDTH_FRAC
    while outer <= _BAND_MAX_OUTER + 1e-9:
        inner = outer - _BAND_WIDTH_FRAC
        angles = _detect_ticks_in_band(ink, lcx, lcy, r, inner, outer)
        score = _band_score(angles)
        if score is not None and score[1] <= _MAX_GAP_COEFF_VAR:
            key = (score[0], -score[1])  # prefer more ticks, then lower variance
            if best_key is None or key > best_key:
                best_key, best_angles = key, angles
        outer += _BAND_STEP

    return best_angles


def _cluster_angles(angles: list, tol_deg: float) -> list:
    """Merge angles within tol_deg of each other (Hough fragments a single
    tick mark into several nearby segments) into one averaged angle each."""
    if not angles:
        return []
    angles = sorted(angles)
    clusters = [[angles[0]]]
    for a in angles[1:]:
        if a - clusters[-1][-1] <= tol_deg:
            clusters[-1].append(a)
        else:
            clusters.append([a])
    if len(clusters) > 1 and (360 - clusters[-1][-1] + clusters[0][0]) <= tol_deg:
        clusters[0] = clusters[-1] + clusters[0]
        clusters.pop()
    return [sum(c) / len(c) for c in clusters]


def _unwrap(angles_sorted: list) -> list:
    """Add 360 wherever needed so angles become monotonically increasing,
    assuming the input is already sorted in physical tick order."""
    out = [angles_sorted[0]]
    for a in angles_sorted[1:]:
        while a < out[-1]:
            a += 360
        out.append(a)
    return out


def _order_ticks_by_gap(angles: list) -> list:
    """Reorder angles to start right after the largest angular gap (the
    gauge's un-marked 'dead zone' between its min and max ticks), then unwrap
    so they're monotonically increasing — recovering physical min-to-max
    tick order without knowing any values yet. Assumes clockwise-increasing
    value, true for the vast majority of pressure/temp/speed gauges."""
    angles = sorted(set(round(a, 3) for a in angles))
    n = len(angles)
    gaps = [(angles[(i + 1) % n] - angles[i]) % 360 for i in range(n)]
    start_idx = (max(range(n), key=lambda i: gaps[i]) + 1) % n
    ordered = angles[start_idx:] + angles[:start_idx]
    return _unwrap(ordered)


_NICE_STEP_MULTIPLIERS = (1, 2, 2.5, 5)
# This check is a pure arithmetic test on (max-min)/(n-1) — it isn't affected
# by angle-measurement noise (a correct detection's step is whatever exact
# value that division produces, given exact min/max and integer n), so a
# tight tolerance can't newly reject a correct detection, only exclude more
# borderline-wrong ones. Tightened from 0.08 after finding it let a real
# wrong case through (5.25 vs "nice" 5 is a 5% deviation — see adaptive tick
# plan.md's images (3).jpg finding).
_NICE_STEP_TOLERANCE = 0.03


def _is_nice_step(step: float) -> bool:
    """Real analog gauges are printed with 'nice' round tick spacing (1, 2,
    2.5, 5 x 10^k — e.g. 10, 20, 0.5, 25). If detected_tick_count doesn't
    match the gauge's actual number of graduations (e.g. one tick missed
    because the needle occludes it), dividing min/max evenly across the
    wrong count produces an ugly, non-round step and a confidently WRONG
    value rather than a safe decline — this check catches that."""
    if step <= 0:
        return False
    magnitude = 10 ** math.floor(math.log10(step))
    return any(abs(step - m * magnitude) / (m * magnitude) <= _NICE_STEP_TOLERANCE for m in _NICE_STEP_MULTIPLIERS)


def build_calibration_ticks(ordered_tick_angles: list, min_value: float, max_value: float) -> list:
    """Assign values to geometrically-ordered tick angles by linear
    interpolation between externally-provided min/max (see module docstring
    for why min/max come from outside this module). Assumes ticks are evenly
    spaced by value, standard for analog gauges. Rejects (returns []) if the
    resulting step size isn't a plausible round number — see _is_nice_step."""
    n = len(ordered_tick_angles)
    if n < 2 or max_value <= min_value:
        return []
    step = (max_value - min_value) / (n - 1)
    if not _is_nice_step(step):
        return []
    return [CalibrationTick(value=min_value + i * step, angle_deg=a) for i, a in enumerate(ordered_tick_angles)]


def _interpolate(needle_angle: float, ticks_sorted: list, unwrapped: list, extrapolate_frac: float = 0.10) -> float | None:
    span = unwrapped[-1] - unwrapped[0]
    if span <= 0:
        return None

    in_range_candidates = [needle_angle + k * 360 for k in (-1, 0, 1) if unwrapped[0] <= needle_angle + k * 360 <= unwrapped[-1]]
    if in_range_candidates:
        a = in_range_candidates[0]
    else:
        a = min((needle_angle + k * 360 for k in (-1, 0, 1)),
                 key=lambda x: min(abs(x - unwrapped[0]), abs(x - unwrapped[-1])))

    tol = span * extrapolate_frac
    if a < unwrapped[0] - tol or a > unwrapped[-1] + tol:
        return None
    a = min(max(a, unwrapped[0]), unwrapped[-1])

    for i in range(len(unwrapped) - 1):
        if unwrapped[i] <= a <= unwrapped[i + 1]:
            if unwrapped[i + 1] == unwrapped[i]:
                continue
            frac = (a - unwrapped[i]) / (unwrapped[i + 1] - unwrapped[i])
            return ticks_sorted[i].value + frac * (ticks_sorted[i + 1].value - ticks_sorted[i].value)
    return None


_MIN_ANGULAR_SWEEP = 30.0


def read_analog_gauge(img_bgr: np.ndarray, min_value: float, max_value: float) -> GaugeReadResult | None:
    """Attempt deterministic CV-based analog gauge reading, calibrated using
    externally-provided scale endpoints (see module docstring for why).
    Returns None if the dial, needle, or tick geometry can't be confidently
    detected — callers should fall back to whatever answer they already had."""
    circle = find_dial_circle(img_bgr)
    if circle is None:
        return None

    needle_angle = find_needle_angle(img_bgr, circle)
    if needle_angle is None:
        return None

    tick_angles = _find_tick_mark_angles(img_bgr, circle)
    if len(tick_angles) < 2:
        return None

    ordered = _order_ticks_by_gap(tick_angles)
    if ordered[-1] - ordered[0] < _MIN_ANGULAR_SWEEP:
        return None

    ticks = build_calibration_ticks(ordered, min_value, max_value)
    if len(ticks) < 2:
        return None

    value = _interpolate(needle_angle, ticks, ordered)
    if value is None:
        return None

    return GaugeReadResult(value=value, circle=circle, needle_angle=needle_angle, ticks_used=ticks)


def _debug_visualize(image_path: str, min_value: float, max_value: float) -> None:
    """Dev-only tool: draws the detected circle/needle/ticks and the ink
    mask, and prints intermediate values.
    Run as: python gauge_reader.py <image_path> --min 0 --max 4"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return

    circle = find_dial_circle(img)
    print(f"circle: {circle}")
    if circle is None:
        cv2.imwrite("gauge_debug.png", img)
        return

    overlay = img.copy()
    cv2.circle(overlay, (int(circle.cx), int(circle.cy)), int(circle.r), (0, 255, 0), 2)
    cv2.circle(overlay, (int(circle.cx), int(circle.cy)), 3, (0, 255, 0), -1)

    x0, y0 = max(int(circle.cx - circle.r), 0), max(int(circle.cy - circle.r), 0)
    x1, y1 = min(int(circle.cx + circle.r), img.shape[1]), min(int(circle.cy + circle.r), img.shape[0])
    ink, _ = _face_ink_mask(img[y0:y1, x0:x1], circle.cx - x0, circle.cy - y0, circle.r)
    cv2.imwrite("gauge_debug_ink_mask.png", ink)
    print("saved ink mask to gauge_debug_ink_mask.png")

    needle_angle = find_needle_angle(img, circle)
    print(f"needle_angle: {needle_angle}")
    if needle_angle is not None:
        rad = math.radians(needle_angle)
        tip_x = int(circle.cx + circle.r * 0.8 * math.cos(rad))
        tip_y = int(circle.cy + circle.r * 0.8 * math.sin(rad))
        cv2.line(overlay, (int(circle.cx), int(circle.cy)), (tip_x, tip_y), (0, 0, 255), 2)

    raw_tick_angles = _find_tick_mark_angles(img, circle)
    print(f"raw tick-mark angles ({len(raw_tick_angles)}): {[round(a, 1) for a in raw_tick_angles]}")

    ordered = _order_ticks_by_gap(raw_tick_angles) if len(raw_tick_angles) >= 2 else []
    ticks = build_calibration_ticks(ordered, min_value, max_value) if ordered else []
    print(f"calibrated ticks: {[(round(t.value, 2), round(t.angle_deg, 1)) for t in ticks]}")
    for t in ticks:
        rad = math.radians(t.angle_deg)
        px = int(circle.cx + circle.r * math.cos(rad))
        py = int(circle.cy + circle.r * math.sin(rad))
        cv2.circle(overlay, (px, py), 4, (255, 0, 0), -1)
        cv2.putText(overlay, f"{t.value:g}", (px + 4, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    result = read_analog_gauge(img, min_value, max_value)
    print(f"read_analog_gauge result: {result}")

    out_path = "gauge_debug.png"
    cv2.imwrite(out_path, overlay)
    print(f"saved overlay to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path")
    parser.add_argument("--min", type=float, default=0.0, dest="min_value")
    parser.add_argument("--max", type=float, default=100.0, dest="max_value")
    args = parser.parse_args()
    _debug_visualize(args.image_path, args.min_value, args.max_value)
