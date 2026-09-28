# claude_vlm_client.py
#
# Claude Sonnet 5 equivalent of vlm_client's three calls, used to compare
# gauge-reading accuracy against the 8B Qwen3-VL/MaintServe backend — see
# gaugesdetectionplan.md for why (the 8B model is inconsistent at analog
# needle-angle reading; a frontier model may do better, though published
# research suggests even frontier VLMs struggle at this specific task, so
# this is an experiment, not an assumed fix).
#
# Reuses vlm_client's prompts and parsing so the comparison is apples-to-
# apples: only the model changes, not the instructions.

import logging

from config import settings
from vlm_client import (
    ENDPOINTS_PROMPT,
    SCALE_RANGE_PROMPT,
    VERIFY_TEXT_PROMPT,
    _ENDPOINTS_ONLY_RE,
    _SCALE_RANGE_RE,
    _strip_thinking,
)

logger = logging.getLogger(__name__)


async def _call_vlm(prompt: str, image_b64: str, zoom_b64: str) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.MAINTSERVE_MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": zoom_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )

    content = "".join(block.text for block in response.content if block.type == "text")
    return _strip_thinking(content)


async def verify_text(image_b64: str, zoom_b64: str, ocr_draft: str) -> str:
    """Send the full image + a zoomed center-crop + OCR draft to Claude Sonnet 5
    and return the corrected, text-only transcription (or best-effort gauge reading)."""
    prompt = VERIFY_TEXT_PROMPT.format(ocr_draft=ocr_draft or "(empty — OCR detected no text)")
    return await _call_vlm(prompt, image_b64, zoom_b64)


async def read_scale_range(image_b64: str, zoom_b64: str) -> tuple:
    """Single-purpose follow-up call: just the scale's min/max and unit."""
    response = await _call_vlm(SCALE_RANGE_PROMPT, image_b64, zoom_b64)
    match = _SCALE_RANGE_RE.match(response.strip())
    if not match:
        return None
    min_value, max_value = float(match.group(1)), float(match.group(2))
    unit = (match.group(3) or "").strip()
    if max_value <= min_value:
        return None
    return (min_value, max_value, unit)


async def read_endpoint_positions(image_b64: str, zoom_b64: str, min_value: float, max_value: float) -> tuple:
    """Single-purpose follow-up call: just the clock positions of the
    already-known min/max labels."""
    prompt = ENDPOINTS_PROMPT.format(min_value=min_value, max_value=max_value)
    response = await _call_vlm(prompt, image_b64, zoom_b64)
    match = _ENDPOINTS_ONLY_RE.match(response.strip())
    if not match:
        return None
    return (float(match.group(1)), float(match.group(2)))
