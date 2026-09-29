# vlm_client.py
#
# Stage 2: vision-model verification, adapted from PrismAPI's
# app/services/maintserve_client.py (same OpenAI-compatible MaintServe/Qwen3-VL
# backend), trimmed to a single call.
#
# Three separate, single-purpose calls rather than one bundled prompt (see
# adaptive tick plan.md): verify_text() classifies the image and gives a
# best-effort fallback reading; read_scale_range() and read_endpoint_positions()
# are narrow, single-question follow-ups used only for analog gauges. This
# split was suggested after finding that a single prompt asking for several
# things at once (reading + scale + endpoint positions) could destabilize an
# otherwise-correct answer on at least one real gauge photo (see
# gaugesdetectionplan.md's regression note) — separating concerns into
# focused calls avoids that cross-talk, at the cost of more round-trips.

import logging
import re

from config import settings

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models."""
    return _THINK_RE.sub("", text).strip()


async def _call_vlm(prompt: str, image_b64: str, zoom_b64: str) -> str:
    """Shared single-call helper: sends the full image + zoomed crop + a text
    prompt, returns the stripped text response. Used by all three VLM calls
    in this module so the client setup isn't duplicated three times."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.MAINTSERVE_BASE_URL,
        api_key="placeholder",  # SDK requires non-empty; actual auth via X-API-Key header
        default_headers={"X-API-Key": settings.MAINTSERVE_API_KEY},
        timeout=120.0,
    )

    response = await client.chat.completions.create(
        model=settings.MAINTSERVE_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{zoom_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=settings.MAINTSERVE_MAX_TOKENS,
        temperature=settings.MAINTSERVE_TEMPERATURE,
    )

    content = response.choices[0].message.content or ""
    return _strip_thinking(content)


VERIFY_TEXT_PROMPT = """\
You are given TWO images of the same subject — the first is the full photo, the \
second is a center-cropped, zoomed-in close-up of it (useful for seeing needle \
position and tick marks in more detail; if the subject is off-center the crop may \
miss part of it, in which case rely on the full image instead) — plus a DRAFT \
transcription produced by an OCR engine on the full image. The OCR draft may \
contain misreadings, garbled characters, wrong word breaks, missing words, or may \
be completely empty (common for analog gauges, since scale numbers are often \
small or embossed and needle position isn't text at all).

OCR DRAFT:
---
{ocr_draft}
---

First decide which of these the image is:

1. GAUGE / METER / DIAL — an instrument currently displaying a measurement (an \
analog dial with a needle/pointer against a printed scale, or a digital LCD/LED \
numeric readout). Give your best-effort reading of the value currently shown \
(for an analog gauge, this is just an estimate — a separate, more precise \
geometric analysis may replace it later) and the unit shown on the gauge \
face/display if visible (e.g. bar, psi, kg/cm2, MPa, °C, A, V, %). On a digital \
LCD/LED readout, look carefully for a decimal point between digits (it's often \
small and easy to miss on a photo) and include it exactly where it appears — \
e.g. a display reading "15.000" is NOT the same as "15000". A real gauge often \
carries OTHER printed text that is not the reading — a manufacturer tag/serial \
plate, brand name, model number, or a safety/usage warning stamped on the \
face — IGNORE all of that entirely and do not transcribe any of it; it is not \
part of the answer no matter how prominent it looks. Output ONLY the reading \
as "<value> <unit>" (e.g. "12 bar"), nothing else on that line and no other \
lines. If multiple separate gauges/readings appear, output one such reading \
per line. The OCR draft cannot see needle position — use it only to help \
confirm the unit label, never as the reading itself.

2. GENERAL TEXT — a document, label, sign, or anything else where the goal is \
transcribing the literal text present.
   - Use the OCR draft as a starting point but correct it against the image — \
fix misread characters and word breaks, add what OCR missed, remove what it \
hallucinated.
   - Output ONLY the text visibly written or printed, exactly as it appears, \
in reading order (top to bottom, left to right).

RULES (both cases):
- Do NOT describe the image, objects, colors, or composition.
- Do NOT add commentary, headings, or explanation of which case you picked.
- Do NOT invent a reading or text that isn't actually visible/inferable.
- If there is truly no text and no gauge reading, output exactly: (no text)

Return ONLY the final reading or text, nothing else.\
"""


SCALE_RANGE_PROMPT = """\
You are given TWO images of the same analog gauge/meter dial — the first is the \
full photo, the second is a center-cropped, zoomed-in close-up (useful for \
seeing small printed numbers more clearly).

Look at the printed scale on the dial face and identify the SMALLEST and \
LARGEST number actually printed on it — its labeled endpoints (e.g. a dial \
marked 0 through 4 has endpoints 0 and 4), regardless of where the needle \
points. Also note the unit shown on the gauge face, if any (e.g. bar, psi, \
kg/cm2, MPa, °C, A, V, %).

This is a plain reading task — read the two endpoint numbers as literal text, \
the same way you'd read any other printed number. Do not describe the needle \
or estimate anything about it.

Output ONLY: <min>-<max> <unit>
e.g.: 0-4 bar

If you cannot clearly read both endpoint numbers, output exactly: unknown\
"""


ENDPOINTS_PROMPT = """\
You are given TWO images of the same analog gauge/meter dial — the first is the \
full photo, the second is a center-cropped, zoomed-in close-up.

This gauge's scale is labeled from {min_value} to {max_value}. Find the printed \
number "{min_value}" and the printed number "{max_value}" on the dial face, and \
report the CLOCK POSITION (1-12, as if viewing the dial like a clock face) \
where EACH of those two specific numbers is printed.

Give each position to the nearest HALF hour — use .5 when a number sits \
between two hour marks. Gauge endpoints very often do (for example a dial \
sweeping from lower-left to lower-right typically starts at 7.5, not 8), and \
rounding those to a whole hour distorts the whole scale.

This is a plain positional reading task — where is a printed number located — \
not a judgment about the needle. Do not describe or estimate the needle's \
position at all.

Output ONLY: <hourMin>-<hourMax>
e.g.: 7.5-4.5

If you cannot clearly judge both positions, output exactly: unknown\
"""


async def verify_text(image_b64: str, zoom_b64: str, ocr_draft: str) -> str:
    """Send the full image + a zoomed center-crop + OCR draft to Qwen3-VL and
    return the corrected, text-only transcription (or best-effort gauge
    reading — see module docstring for why scale/endpoint details are
    separate calls, not part of this one)."""
    prompt = VERIFY_TEXT_PROMPT.format(ocr_draft=ocr_draft or "(empty — OCR detected no text)")
    return await _call_vlm(prompt, image_b64, zoom_b64)


_SCALE_RANGE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*([^\d\s].*)?\s*$")


async def read_scale_range(image_b64: str, zoom_b64: str) -> tuple:
    """Single-purpose follow-up call: just the scale's min/max and unit, for
    analog gauges. Returns (min_value, max_value, unit) or None."""
    response = await _call_vlm(SCALE_RANGE_PROMPT, image_b64, zoom_b64)
    match = _SCALE_RANGE_RE.match(response.strip())
    if not match:
        return None
    min_value, max_value = float(match.group(1)), float(match.group(2))
    unit = (match.group(3) or "").strip()
    if max_value <= min_value:
        return None
    return (min_value, max_value, unit)


_ENDPOINTS_ONLY_RE = re.compile(r"^\s*(\d{1,2}(?:\.\d+)?)\s*-\s*(\d{1,2}(?:\.\d+)?)\s*$")


async def read_endpoint_positions(image_b64: str, zoom_b64: str, min_value: float, max_value: float) -> tuple:
    """Single-purpose follow-up call: just the clock positions of the
    already-known min/max labels (see ENDPOINTS_PROMPT — deliberately asks
    about the far-apart scale endpoints, not the ticks immediately
    bracketing the needle, and deliberately never mentions where the needle
    actually is — see adaptive tick plan.md for why both of those choices
    matter). Returns (hour_min, hour_max) or None."""
    prompt = ENDPOINTS_PROMPT.format(min_value=min_value, max_value=max_value)
    response = await _call_vlm(prompt, image_b64, zoom_b64)
    match = _ENDPOINTS_ONLY_RE.match(response.strip())
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))
