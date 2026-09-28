# Plan: Deterministic OpenCV needle-angle detection for analog gauges

## Context

The gauge-reading feature added earlier (a prompt change in `vlm_client.py` asking the 8B Qwen3-VL model to interpret analog needle position and digital displays) works for digital displays but is unreliable for analog needles: tested against a real gauge photo where the needle visually points almost exactly at "1" on a 0–4 bar scale, the model confidently answered "0.5 bar" — unchanged even after adding chain-of-thought prompting and a zoomed crop (both confirmed ineffective: the model produces no visible `<think>` reasoning trace at all, and the zoom crop was correctly framed but didn't change the answer). Only an 8B model is available on this backend, so there's no bigger-model lever to pull. Visually interpolating a needle's angle against scale ticks is a genuine capability gap of this model, not a prompting or image-quality problem.

The fix is to stop asking the VLM to eyeball the needle angle and instead compute it deterministically: find the dial with a Hough Circle Transform, find the needle with a Hough Line Transform, calibrate the scale using OCR-detected tick numbers' angular positions, and interpolate. This is the standard approach for this class of problem, validated against known prior art. It's confirmed no reusable gauge-reading code exists anywhere on this machine (checked earlier, including the PrismAPI-derived origins of this project) — this is new code.

**Decisions made:**
- CV output only ever **overrides the numeric value** of an answer the VLM has *already* classified as a single gauge reading (its output already matches `"<number> <unit>"` on one line) — it never injects a reading into what the VLM decided was plain text or a multi-gauge image. This keeps the VLM in charge of classification; CV only improves precision once gauge-ness is already established.
- Every detection stage must be able to cleanly return "unknown" rather than guess — no circle found, no needle found, fewer than 2 calibration ticks, needle angle outside the calibrated range, or too-short an angular sweep to trust → the whole pipeline returns `None` and the existing VLM answer is kept as-is.
- Multi-gauge images and dual-scale gauges (e.g. one face with both psi and bar numbers) are explicitly out of scope for v1 — both are detected defensively and cause the CV path to back off to `None` rather than produce a corrupted reading.
- Include a debug visualizer script (for tuning) and a small pytest regression suite against the 3 existing sample photos, since this project currently has zero test infrastructure.

## Approach

### 1. `ocr_engine.py` — expose OCR bounding boxes (minimal refactor)

`extract_draft_text()` currently discards bounding boxes from RapidOCR's result, keeping only joined text. The gauge calibration step needs per-detection boxes to compute each tick number's angle from the dial center.

- Change `_preprocess()` to return `(processed_img, scale_factor)` instead of just the image — `scale_factor` is the named constant used for the 2x upscale (`_OCR_UPSCALE_FACTOR = 2`) on the normal path, or `1.0` on its existing exception-fallback path (currently it silently falls back to the *original* image on preprocessing failure; today that's harmless since only text is kept, but a bbox-returning function must know which scale actually applied, or boxes get silently corrupted on that fallback path).
- Add `detect_text(img) -> list[OcrDetection]` (a small `@dataclass OcrDetection(bbox, text, conf)`) that runs the engine and rescales each bbox back into `img`'s own coordinate space using the returned scale factor.
- Rewrite `extract_draft_text()` as a thin wrapper: `" ".join(d.text for d in detect_text(img))` — output stays byte-for-byte identical to today, so this is a safe, additive refactor.

### 2. `gauge_reader.py` (new) — the CV pipeline

Pure OpenCV + math, no FastAPI/VLM dependency, so it's independently testable. Angle convention used throughout: `angle = atan2(y - cy, x - cx) % 360` in raw image pixel coordinates (0° = 3 o'clock, 90° = 6 o'clock, clockwise-increasing) — document this prominently since a sign error here silently produces backwards-but-plausible-looking readings.

**Step 0 — resolution normalization**: downscale (via `cv2.resize`, `INTER_AREA`) to a max dimension of ~1000px before circle/needle detection so fixed-pixel kernel parameters behave consistently across arbitrary photo sizes; angles are scale-invariant so this doesn't affect the final answer. The OCR calibration step instead crops from the *original* full-resolution image (mapping the circle back up by `1/scale`) since downscaling would blur small tick digits further.

**Step 1 — `find_dial_circle(img) -> DialCircle | None`**: grayscale + median blur, then `cv2.HoughCircles` (`dp=1.5`, `param1=100`, `param2=60`, `minRadius=0.15×min(h,w)`, `maxRadius=0.48×min(h,w)`, `minDist=0.5×min(h,w)`), picking the largest returned circle. On no match, retry once with a looser `param2=40` and widened radius range before giving up. `param2`, `minRadius`/`maxRadius` are the highest-risk parameters (gauge framing varies a lot) — expect to retune against real photos.

**Step 2 — `find_needle_angle(img, circle) -> float | None`**: crop to the dial ROI, Canny edges masked to the inner 95% of the circle (drops bezel/casing edges), `cv2.HoughLinesP` (`minLineLength=0.3×r`, `maxLineGap=0.05×r`). Filter candidate segments to those with one endpoint within `0.15×r` of the center (the pivot) and the other at least `0.35×r` out (reaching toward the rim) — this is what distinguishes the needle from tick marks, which live near the rim and don't pass near center. Needles fragment into several short collinear segments under Canny; v1 picks the longest qualifying segment (simple, acceptable starting point — upgrade to angle-clustering if testing shows jitter). On zero candidates, retry once with looser Canny thresholds before giving up (thin/light-colored needles may not produce strong edges at the first threshold pair).

**Step 3 — `get_calibration_ticks(img, circle) -> list[CalibrationTick]`**: crop a margin around the dial (`1.4×r`, from the original full-res image) and run `ocr_engine.detect_text()` on it. Keep only detections matching a numeric regex, restrict to those whose bbox centroid falls in a `0.4r–1.35r` radius band from center (excludes brand text/model numbers near the middle), compute each kept tick's angle. Add an outlier filter: compute each consecutive value-sorted pair's degrees-per-unit ratio and drop ticks whose ratio deviates wildly from the median — cheaply catches stray numeric text (serial numbers) that happens to land in the radius band. Require ≥2 ticks after filtering.

**Step 4 — interpolate**: sort ticks by their *parsed numeric value*, unwrap angles by adding 360° wherever needed to keep them monotonically increasing (this sidesteps figuring out geometrically which side is "start" — assumes clockwise-increasing gauges, true for the vast majority of pressure/temp/speed gauges; document as a limitation). Map the needle's angle into the same winding, requiring it fall within the tick range plus a 10% extrapolation tolerance (clamped to the boundary) — outside that, return `None` rather than guess. Also gate on a minimum angular sweep across the calibrated ticks (~30°) — too short a sweep means ordinary angle-measurement noise translates into large value error.

**Top-level `read_analog_gauge(img) -> GaugeReadResult | None`**: chains all four steps, returning `None` immediately at any gate failure (no circle, no needle, <2 ticks, sweep too short, needle out of range).

### 3. `main.py` — integration

After the existing VLM call succeeds (`verified == True`), check whether `final_text` is a single line matching `"<number> <unit>"`. Only then call `gauge_reader.read_analog_gauge(img)` (reusing the already-decoded full-resolution `img` from the OCR stage — no re-decoding needed), wrapped in `try/except` so a bug in the brand-new CV module can never break the endpoint. If it returns a result, replace just the numeric portion of `final_text` with the CV-computed value (formatted to 2 decimals, trailing zeros stripped), keeping the VLM's unit. Add a `"cv_override": bool` field to the response (additive, non-breaking) so it's visible from live traffic whether the CV path fired, without needing to grep logs.

### 4. Known limitations to document (in code comments, not user-facing)

Dual-scale gauges (two number sets on one face) can corrupt calibration if both scales' ticks fall in the same radius band — mitigate by checking for a bimodal radius distribution among candidate ticks and bailing to `None` if detected, rather than attempting to disambiguate which scale is meant. Non-circular/oval dials from extreme camera angles aren't corrected (no perspective rectification in v1). OCR digit confusions (O/0, l/1) can silently drop or corrupt individual ticks — not fully fixable by regex alone. Multiple needles (e.g. a drag/min-max indicator) resolve to whichever produces the longest qualifying segment.

### 5. Testing

- **Debug visualizer**: `python gauge_reader.py <image_path>` (dev-only entry point, not wired into the FastAPI app) — draws the detected circle, needle line, and accepted calibration tick boxes+values onto a copy of the image, saves it, and prints all intermediate values (circle params, needle angle, tick list, interpolated value). This is the primary tool for empirically tuning the parameters above against real photos — use it first, before touching `main.py` integration.
- **Regression tests**: add `pytest` to `requirements.txt`, create `tests/fixtures/` with the 3 existing sample images (`images.jpg`, `images (1).jpg`, `images (2).jpg` from the Desktop), and `tests/test_gauge_reader.py` asserting: the digital-display sample (`images (2).jpg`) returns `None` from `read_analog_gauge`; the two analog samples return a value within a reasonable tolerance (±0.2–0.3) of their known/eyeballed readings — in particular confirming the "needle at ~1, VLM said 0.5" case now resolves near 1.

## Files to change
- `image_script/image_script/ocr_engine.py` — expose bboxes via new `detect_text()`, keep `extract_draft_text()` behavior identical.
- `image_script/image_script/gauge_reader.py` — new module, the full CV pipeline described above.
- `image_script/image_script/main.py` — call the pipeline after a successful gauge-shaped VLM answer, merge results, add `cv_override` to the response.
- `image_script/image_script/requirements.txt` — add `pytest`.
- `image_script/image_script/tests/test_gauge_reader.py` (new) + `tests/fixtures/` (new, copies of the 3 sample images).

## Verification
1. Run the debug visualizer against all 3 sample images first, tuning Hough/Canny/radius-band parameters until the drawn overlays look correct by eye (circle on the dial, line on the needle, boxes on the real scale numbers) — do this before wiring anything into `main.py`.
2. Run `pytest` — digital sample returns `None`, both analog samples land within tolerance of their real readings.
3. Restart the server, re-run the same `curl -F "file=@..."` tests used earlier against all 3 sample images plus the plain-text regression image, confirming: `images (1).jpg` now reads close to 1 bar (not 0.5), the digital sample is unaffected (CV correctly declines, VLM's literal-text answer is used untouched), and plain-text extraction still works with `cv_override: false`.
