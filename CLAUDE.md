# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small FastAPI service (`image_script/main.py`) with a single real endpoint: upload an image, get back the text found in it. It does two-stage extraction, adapted from a larger project called PrismAPI:

1. **OCR draft** (`ocr_engine.py`) — RapidOCR runs locally on the image (CLAHE contrast enhancement + 2x upscale preprocessing) to produce a fast, rough transcription.
2. **VLM verification** (`vlm_client.py`) — the original image plus the OCR draft are sent to a remote Qwen3-VL model (via an OpenAI-compatible "MaintServe" backend) with a prompt instructing it to correct the draft against what it actually sees, returning text only (no image descriptions).

If the VLM call fails or is unreachable, `main.py` falls back to returning the raw OCR draft so the endpoint stays usable, and reports `"verified": false`.

All code lives in the `image_script/` subdirectory of the repo root.

## Commands

Run from the `image_script/` directory.

Install dependencies:
```
pip install -r requirements.txt
```

Run the server:
```
python main.py
```
or
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

There is no lint, test, or build tooling configured in this repo.

Manual check once running:
```
curl http://localhost:8000/health
curl -F "file=@some_image.png" http://localhost:8000/extract-text
```

## Configuration

Settings are loaded via `pydantic-settings` in `config.py` from a `.env` file (see `.env.example`) in `image_script/`:

- `MAINTSERVE_BASE_URL` — OpenAI-compatible base URL for the Qwen3-VL backend.
- `MAINTSERVE_API_KEY` — sent as the `X-API-Key` header (not as the OpenAI SDK's bearer token — the SDK's own `api_key` field is a required placeholder and unused for auth).
- `MAINTSERVE_MODEL`, `MAINTSERVE_MAX_TOKENS`, `MAINTSERVE_TEMPERATURE` — model call parameters.

## Architecture notes

- `main.py` is the only HTTP surface: `/health` and `POST /extract-text`. It validates content-type (`ALLOWED_CONTENT_TYPES`), decodes the upload twice — once via OpenCV/numpy for the OCR stage, once via Pillow to re-encode as JPEG for the VLM stage — since the two stages need different image representations.
- The OCR and VLM stages are independent modules with no shared state; `main.py` orchestrates them and owns the fallback behavior. Each stage catches its own exceptions internally so a failure in one doesn't take down the other.
- `ocr_engine.py` lazily initializes a module-level `RapidOCR` singleton (`_get_engine()`) since engine construction is expensive; `rapidocr_onnxruntime` is imported defensively (`RapidOCR = None` if missing) so the module can still be imported without the dependency installed.
- `vlm_client.py`'s `VERIFY_TEXT_PROMPT` is the core behavioral contract for stage 2 — it constrains the VLM to literal transcription only (no scene/object descriptions) and defines the `(no text)` sentinel for text-free images. Changing extraction behavior usually means editing this prompt. `_strip_thinking()` strips `<think>...</think>` blocks since the backing model may be a reasoning model.
