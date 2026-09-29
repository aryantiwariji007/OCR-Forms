# main.py
#
# Small FastAPI script: image in, text out.
#
# Two-stage extraction, inspired by PrismAPI:
#   Stage 1 (ocr_engine.py)  — RapidOCR produces a fast local draft transcription.
#   Stage 2 (vlm_client.py)  — Qwen3-VL looks at the actual image plus that draft
#                              and returns a corrected, text-only transcription
#                              (no image/object descriptions — see VERIFY_TEXT_PROMPT).
# If the VLM call fails, the endpoint falls back to the OCR draft so the script
# stays usable even when the remote vision backend is unreachable.

import base64
import io
import logging
import re

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

import claude_vlm_client
import digital_display_reader
import gauge_reader
import image_quality
from config import settings
from ocr_engine import extract_draft_text
from vlm_client import read_endpoint_positions, read_scale_range, verify_text

# Matches a single-line VLM answer already shaped like a gauge reading, e.g.
# "12 bar" or "0.5". Only in that case do we attempt to override the number
# with a deterministic CV reading — this keeps the VLM in charge of deciding
# whether an image is a gauge at all; CV only refines precision once that's
# already established. See gaugesdetectionplan.md.
_GAUGE_LINE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([^\d\s].*)?\s*$")

# CV is meant to REFINE the VLM's own reading, not contradict it outright —
# a large divergence is a stronger signal that something upstream is wrong
# (e.g. the VLM misread the scale's min/max on an ambiguous dual-scale gauge)
# than that CV found the "true" value. Cap how far the CV override is allowed
# to move the answer, as a fraction of the reported scale's own span.
_MAX_CV_DIVERGENCE_FRAC = 0.25


def _format_value(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


# Calibrating a needle angle into a value needs two plain reading tasks — what
# numbers are printed at the ends of the scale, and where those numbers sit.
# Both are literal transcription rather than spatial estimation, which is the
# kind of reading Claude has been consistently better at here (it read a
# -1..5 bar compound scale correctly where the primary VLM reported 0..5,
# wrongly, and the bad range made the whole calibration unusable). The primary
# VLM remains the fallback so the pipeline still works with no Anthropic key
# configured. Note this is the opposite of judging the needle itself, where
# Claude was NOT more reliable — that judgement stays with our own geometry.
async def _read_scale_range_best(image_b64: str, zoom_b64: str):
    if settings.ANTHROPIC_API_KEY:
        try:
            result = await claude_vlm_client.read_scale_range(image_b64, zoom_b64)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("Claude scale-range call failed, falling back: %s", e)
    return await read_scale_range(image_b64, zoom_b64)


async def _read_endpoints_best(image_b64: str, zoom_b64: str, min_value: float, max_value: float):
    if settings.ANTHROPIC_API_KEY:
        try:
            result = await claude_vlm_client.read_endpoint_positions(image_b64, zoom_b64, min_value, max_value)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("Claude endpoint-position call failed, falling back: %s", e)
    return await read_endpoint_positions(image_b64, zoom_b64, min_value, max_value)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Text Extractor")

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff",
}


def _center_crop_zoom(pil_img: Image.Image, crop_fraction: float = 0.7, upscale: float = 1.5) -> Image.Image:
    """Center-crop to crop_fraction of each dimension, then upscale — a cheap
    "zoom in on the middle" with no per-image calibration."""
    w, h = pil_img.size
    cw, ch = int(w * crop_fraction), int(h * crop_fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = pil_img.crop((left, top, left + cw, top + ch))
    return cropped.resize((int(cw * upscale), int(ch * upscale)), Image.LANCZOS)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract-text")
async def extract_text(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
                    f"Supported: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")

    # Decode for OCR
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    # Stage 1: local OCR draft
    try:
        ocr_draft = extract_draft_text(img)
    except Exception as e:
        logger.error("OCR stage failed: %s", e)
        ocr_draft = ""

    # Dial/needle detection, done BEFORE the VLM call: unlike tick-mark
    # calibration (which repeatedly fails on low-resolution photos — see
    # adaptive tick plan.md), needle-angle detection alone has been reliably
    # correct on every test photo this session. Reused below for the
    # sequential scale/endpoint calls and the post-VLM CV-calibration
    # attempt, so the dial is only detected once per request.
    try:
        circle = gauge_reader.find_dial_circle(img)
        needle_angle = gauge_reader.find_needle_angle(img, circle) if circle is not None else None
    except Exception as e:
        logger.warning("Pre-VLM dial/needle detection failed: %s", e)
        circle = None
        needle_angle = None

    # Stage 2: VLM verification against the actual image
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Center-cropped, upscaled close-up — gives the VLM more effective pixels on
    # needle/tick detail for gauge readings. Sent alongside the full image since
    # off-center subjects would otherwise get clipped by the crop.
    zoom_buf = io.BytesIO()
    _center_crop_zoom(pil_img).save(zoom_buf, format="JPEG", quality=92)
    zoom_b64 = base64.b64encode(zoom_buf.getvalue()).decode("ascii")

    try:
        final_text = await verify_text(image_b64, zoom_b64, ocr_draft)
        verified = True
    except Exception as e:
        logger.warning("VLM verification failed, falling back to OCR draft: %s", e)
        final_text = ocr_draft
        verified = False

    # The model sometimes returns more than just the reading — an explanatory
    # paragraph after a blank line despite being told not to
    # ("5.253\n\nThe image is a digital gauge, as evidenced by..."), or
    # incidental text printed on the gauge's own face transcribed as extra
    # lines even though the prompt now says to ignore it (a tag plate, brand
    # name, or safety warning, e.g. "0.8 kg/cm2\nPRESSURE GAUGE", or several
    # such lines on a heavily-labeled industrial gauge). When the FIRST line
    # alone already looks like a complete gauge reading, trust only that line
    # and discard everything after it, regardless of how it's separated — a
    # real reading is never legitimately followed by more prose, and text
    # printed elsewhere on the gauge's face doesn't change what the needle is
    # pointing at. This keeps the VLM in charge of deciding whether an image
    # is a gauge at all (see _GAUGE_LINE_RE) — CV below only refines
    # precision once that's already established from the model's own first
    # line. A genuine multi-line document is virtually never a bare
    # "<number> <unit>" line followed by more content, so this doesn't
    # meaningfully risk truncating real text.
    if verified:
        first_line = next((ln.strip() for ln in final_text.strip().splitlines() if ln.strip()), "")
        if _GAUGE_LINE_RE.match(first_line):
            final_text = first_line

    # Analog-gauge precision refinement: the VLM is unreliable at visually
    # interpolating a needle's angle (see gaugesdetectionplan.md), so when its
    # answer already looks like a single gauge reading, try to replace just
    # the number with a deterministic CV-computed one. Never invoked when the
    # VLM decided this was plain text or reported multiple readings.
    cv_applied = False
    quality_warning = None
    if verified:
        lines = final_text.strip().splitlines()
        if len(lines) == 1:
            match = _GAUGE_LINE_RE.match(lines[0])
            if match:
                # Digital-display precision refinement first: a classical
                # 7-segment decoder is inherently more reliable than a VLM
                # reading a compressed photo of an LED/LCD display — it lost
                # the decimal point repeatedly in testing (see adaptive tick
                # plan.md). Needs no VLM-provided calibration, unlike the
                # analog CV path below, so no divergence check either — a
                # validated segment-pattern match isn't a guess the way an
                # analog needle-angle interpolation can be.
                try:
                    digital_result = digital_display_reader.read_digital_display(img)
                except Exception as e:
                    logger.warning("Digital display decode failed: %s", e)
                    digital_result = None

                if digital_result is not None:
                    unit = (match.group(2) or "").strip()
                    final_text = f"{digital_result.text} {unit}".strip()
                    cv_applied = True
                    logger.info("Digital display decode applied: %s", digital_result.text)
                else:
                    # Sequential follow-up calls (see adaptive tick plan.md
                    # for why separate, single-purpose calls rather than one
                    # bundled prompt): scale range is always worth asking for
                    # analog gauges (it also feeds the tick-based CV fallback
                    # below); endpoint clock positions are only useful when
                    # we have our own reliable needle_angle to interpolate
                    # against.
                    try:
                        scale_result = await _read_scale_range_best(image_b64, zoom_b64)
                    except Exception as e:
                        logger.warning("Scale-range call failed: %s", e)
                        scale_result = None

                    min_value = max_value = scale_unit = None
                    if scale_result is not None:
                        min_value, max_value, scale_unit = scale_result

                    # Actionable feedback for the caller when the photo itself is
                    # likely too small/blurry for reliable analog reading — most
                    # CV/OCR failures this session traced back to insufficient
                    # pixel information in the source photo, not algorithm choice
                    # (see gaugesdetectionplan.md / adaptive tick plan.md).
                    quality_warning = image_quality.assess_gauge_quality(img, circle)

                    endpoints = None
                    if min_value is not None and max_value is not None and needle_angle is not None:
                        try:
                            endpoints = await _read_endpoints_best(image_b64, zoom_b64, min_value, max_value)
                        except Exception as e:
                            logger.warning("Endpoint-position call failed: %s", e)
                            endpoints = None

                    # Endpoint-based interpolation: the VLM reported the
                    # clock positions of the scale's min/max labels (a plain
                    # reading task, no judgment about the needle) and WE
                    # compute the value deterministically from our own
                    # precise needle_angle, rather than trusting whatever
                    # value the model rendered itself in Call 1. Tried first
                    # since it's more targeted than full tick-mark
                    # calibration below, and doesn't need tick detection to
                    # succeed at all.
                    endpoints_applied = False
                    if endpoints is not None:
                        hour_min, hour_max = endpoints
                        try:
                            endpoint_value = gauge_reader.interpolate_from_two_anchors(
                                needle_angle, min_value, hour_min, max_value, hour_max
                            )
                        except Exception as e:
                            logger.warning("Endpoint interpolation failed: %s", e)
                            endpoint_value = None
                        if endpoint_value is not None:
                            vlm_value = float(match.group(1))
                            span = max_value - min_value
                            too_divergent = abs(endpoint_value - vlm_value) > span * _MAX_CV_DIVERGENCE_FRAC
                            if too_divergent:
                                logger.warning(
                                    "Endpoint-interpolated value diverges too much from the VLM's own "
                                    "reading (endpoint=%.3g vlm=%.3g) — discarding",
                                    endpoint_value, vlm_value,
                                )
                            else:
                                unit = (match.group(2) or scale_unit or "").strip()
                                final_text = f"{_format_value(endpoint_value)} {unit}".strip()
                                cv_applied = True
                                endpoints_applied = True
                                logger.info(
                                    "Endpoint-based interpolation applied: value=%.3g endpoints=(%g@%g,%g@%g)",
                                    endpoint_value, min_value, hour_min, max_value, hour_max,
                                )

                    if not endpoints_applied and circle is not None and min_value is not None and max_value is not None:
                        try:
                            cv_result = gauge_reader.read_analog_gauge(img, min_value=min_value, max_value=max_value)
                        except Exception as e:
                            logger.warning("Gauge CV pipeline failed: %s", e)
                            cv_result = None
                        if cv_result is not None:
                            vlm_value = float(match.group(1))
                            max_divergence = (max_value - min_value) * _MAX_CV_DIVERGENCE_FRAC
                            if abs(cv_result.value - vlm_value) > max_divergence:
                                logger.warning(
                                    "Gauge CV result diverges too much from the VLM's own reading "
                                    "(cv=%.3g vlm=%.3g max_allowed_diff=%.3g) — discarding CV override",
                                    cv_result.value, vlm_value, max_divergence,
                                )
                            else:
                                unit = (match.group(2) or scale_unit or "").strip()
                                final_text = f"{_format_value(cv_result.value)} {unit}".strip()
                                cv_applied = True
                                logger.info(
                                    "Gauge CV override applied: needle_angle=%.1f value=%.3g ticks=%s",
                                    cv_result.needle_angle, cv_result.value,
                                    [round(t.value, 3) for t in cv_result.ticks_used],
                                )

                    # Last-resort refinement: nothing classical corroborated the
                    # VLM's number. Empirically, the primary VLM (Qwen) reliably
                    # drops the decimal point on digital LCD/LED readouts
                    # ("15.000"->"15000", "9.375"->"9375" — confirmed on multiple
                    # real photos), while Claude Sonnet 5 reads the same photos'
                    # decimal correctly every time tested. That's the opposite of
                    # analog needle-ANGLE judgment, where Claude was previously
                    # found NOT more reliable (and worse on at least one
                    # needle-at-rest case — see claude_vlm_client.py's module
                    # docstring): reading a printed digital display is literal
                    # transcription, not fine spatial estimation, and frontier
                    # models are comparatively much stronger at the former. Only
                    # consulted here, after every classical path has already
                    # failed to apply, so this can never override an
                    # already-correct CV result and can't reintroduce that
                    # analog regression.
                    if not cv_applied and settings.ANTHROPIC_API_KEY:
                        try:
                            claude_text = await claude_vlm_client.verify_text(image_b64, zoom_b64, ocr_draft)
                        except Exception as e:
                            logger.warning("Claude refinement call failed: %s", e)
                            claude_text = None
                        if claude_text:
                            claude_first_line = next(
                                (ln.strip() for ln in claude_text.strip().splitlines() if ln.strip()), ""
                            )
                            if _GAUGE_LINE_RE.match(claude_first_line):
                                final_text = claude_first_line
                                logger.info("Claude refinement applied: %s", final_text)

                    # None of the classical paths above (digital 7-segment decode,
                    # endpoint interpolation, tick-based calibration) actually
                    # corroborated the number — assess_gauge_quality's
                    # circle-radius/blur checks don't catch every such case (e.g.
                    # a circular gauge HOUSING gets detected as a "dial circle"
                    # even for a digital display with no needle at all, so its
                    # checks can pass while the digital decoder still silently
                    # declined — seen empirically). Surface that explicitly rather
                    # than returning an unverified guess with no warning at all.
                    if not cv_applied and quality_warning is None:
                        quality_warning = (
                            "Could not independently verify this reading with classical "
                            "detection — treat this value as an estimate."
                        )

    return {"text": final_text, "verified": verified, "cv_override": cv_applied, "quality_warning": quality_warning}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
