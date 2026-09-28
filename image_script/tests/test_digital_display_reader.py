# test_digital_display_reader.py
#
# Regression test for digital_display_reader.py against the real OMEGA
# digital gauge fixture (see adaptive tick plan.md). The true displayed
# value is "5.253" — confirmed by both zoomed visual inspection of the
# source photo and this decoder's own segment-level decode.

import os

import cv2

from digital_display_reader import read_digital_display

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _load(name):
    path = os.path.join(FIXTURES_DIR, name)
    img = cv2.imread(path)
    assert img is not None, f"failed to load fixture: {path}"
    return img


def test_digital_gauge_decodes_correctly():
    img = _load("gauge_digital_1.jpg")
    result = read_digital_display(img)
    assert result is not None, "expected a confident decode on this clear fixture"
    assert result.text == "5.253"


def test_analog_gauge_photos_decline():
    """No 7-segment display exists on these analog dial photos — the
    decoder must not fabricate a reading for them."""
    for name in ("gauge_analog_1.jpg", "gauge_analog_2.jpg"):
        img = _load(name)
        assert read_digital_display(img) is None, f"{name}: expected None (no digital display present)"
