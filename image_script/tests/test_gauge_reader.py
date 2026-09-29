# test_gauge_reader.py
#
# Regression suite for gauge_reader.py against the 3 real sample photos used
# during development (see gaugesdetectionplan.md / adaptive tick plan.md).
# read_analog_gauge() is designed to cleanly return None rather than guess
# when it isn't confident — as of this writing, calibration doesn't yet
# succeed on any of these 3 real photos (small/low-res close-ups defeat
# tick-mark detection, or the detected tick count doesn't match a plausible
# round step size), so these tests currently just pin that safe "decline"
# behavior and act as a tripwire: if future tuning makes calibration start
# succeeding on one of them, the tolerance check kicks in automatically.
#
# min_value/max_value below are the real printed scale endpoints on each
# gauge (what a caller would get from the VLM reading them as plain text —
# see gauge_reader.py's module docstring for why calibration takes these as
# external input rather than reading them itself).

import os

import cv2
import pytest

from gauge_reader import find_dial_circle, find_needle_angle, read_analog_gauge

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")

ANALOG_FIXTURES = [
    # (filename, min_value, max_value, expected, tolerance)
    ("gauge_analog_1.jpg", 0.0, 60.0, 10.0, 5.0),   # Parker dial, psi scale, needle near "10"
    ("gauge_analog_2.jpg", 0.0, 4.0, 1.0, 0.3),
]


def _load(name):
    path = os.path.join(FIXTURES_DIR, name)
    img = cv2.imread(path)
    assert img is not None, f"failed to load fixture: {path}"
    return img


def test_digital_display_returns_none():
    """No physical needle exists on a digital LCD readout — the pipeline
    must not fabricate an analog reading for it."""
    img = _load("gauge_digital_1.jpg")
    assert read_analog_gauge(img, min_value=0.0, max_value=100.0) is None


def test_needle_points_at_tip_not_counterweight():
    """The Parker dial carries a stubby counterweight opposite a long thin
    pointer. Reading the counterweight end put an earlier version ~180 degrees
    out, so pin the direction: the tip sits down-left toward "0" (~133 deg in
    this module's convention, where 0 deg is 3 o'clock and angles run
    clockwise), NOT up-right at ~313 deg."""
    img = _load("gauge_analog_1.jpg")
    circle = find_dial_circle(img)
    assert circle is not None
    angle = find_needle_angle(img, circle)
    assert angle is not None
    assert abs(angle - 133.0) < 15.0, f"expected tip near 133 deg, got {angle}"


def test_needle_with_reflective_highlight_still_resolves():
    """This gauge's needle is glossy enough to carry a bright specular
    highlight down its own centre, which used to make the ink mask see only
    the needle's two edges (a hollow outline) instead of one solid shape —
    Hough couldn't trace a line through that at all and locked onto unrelated
    noise (bezel rust, label text) instead. It also has an unusual needle
    shape (a diamond-shaped counterweight opposite an arrowhead-flared tip)
    that the original fixed-radius thickness check couldn't tell apart either
    (both sides measured identically at 0.20-0.36r). Pin the fixed behavior:
    tip resolves to ~302 deg (pointing toward "4000" on the scale), not ~122
    deg (the diamond counterweight toward "0")."""
    img = _load("gauge_analog_3_reflective_needle.jpg")
    circle = find_dial_circle(img)
    assert circle is not None
    angle = find_needle_angle(img, circle)
    assert angle is not None
    assert abs(angle - 302.0) < 15.0, f"expected tip near 302 deg, got {angle}"


def test_picks_the_dial_circle_over_a_larger_spurious_one():
    """Hough found three circle candidates on this photo: the real dial, plus
    two spurious ones from bezel/tag-plate texture — one of which was larger
    by a narrow margin and extended almost an entire radius past the bottom of
    the frame. Picking "largest radius" alone chose that wrong one and put the
    circle's centre near the tag plate instead of the pivot, ~280px off (0.98r
    of its own radius) from the true dial. Pin that the real, mostly-onscreen
    dial wins instead."""
    img = _load("gauge_analog_4_bad_circle_pick.jpg")
    circle = find_dial_circle(img)
    assert circle is not None
    assert abs(circle.cx - 284) < 20 and abs(circle.cy - 253) < 20, (
        f"expected centre near (284, 253), got ({circle.cx}, {circle.cy})"
    )


def test_declines_when_circle_centre_misses_the_hub():
    """This fixture's dial circle fits badly — its centre lands ~0.21r from the
    real pivot, which skewed the measured angle by roughly 30 degrees while
    still looking plausible. Angles are only meaningful when measured from the
    true pivot, so detection must decline rather than return that."""
    img = _load("gauge_analog_2.jpg")
    circle = find_dial_circle(img)
    assert circle is not None
    assert find_needle_angle(img, circle) is None


@pytest.mark.parametrize("filename,min_value,max_value,expected,tolerance", ANALOG_FIXTURES)
def test_analog_gauge_no_crash_and_plausible_if_confident(filename, min_value, max_value, expected, tolerance):
    img = _load(filename)
    result = read_analog_gauge(img, min_value=min_value, max_value=max_value)  # must not raise
    if result is not None:
        assert abs(result.value - expected) <= tolerance, (
            f"{filename}: got {result.value}, expected ~{expected} (+/- {tolerance})"
        )
