# Plan: Adaptive tick-band search + ellipse rectification + upload quality feedback

## Context

`gauge_reader.py` currently uses fixed fractional constants (`_TICK_BAND_INNER=0.55`, `_TICK_BAND_OUTER=0.92`) to search for tick marks near the rim. Testing this session showed this doesn't generalize: on `images (1).jpg` it found nothing (ticks too faint at this photo's resolution), and on `images.jpg` it locked onto the **outer chrome bezel** instead of the real printed scale, because that gauge has a proportionally wider bezel than the close-up bar gauge the constants were tuned against. A single fixed band can't work across differently-proportioned gauge photos. Needle-angle detection, by contrast, has been reliable across every test this session — the problem is specifically in tick-mark *localization*.

Separately, comparison testing (Qwen 8B vs Claude Sonnet 5) and the earlier OCR investigation both point to the same root cause across nearly every failure mode hit this session: **insufficient pixel information in small/close-up/blurry source photos** — not a fixable algorithm choice. Since that's outside this codebase's control per-photo, the practical lever is giving the *caller* actionable feedback so future uploads are better, rather than silently returning a low-confidence answer.

This plan tackles both remaining items from the options list: (1) classical CV tuning — replace the fixed tick band with a per-image adaptive search, and add ellipse rectification for off-axis photos; (2) upload quality feedback — surface a warning when a photo's detected dial is too small or too blurry for reliable analog reading.

**Honest expectation-setting:** this is the fourth attempt at fixing tick-mark detection this session. Each prior attempt fixed one failure mode and hit a different one. This plan is worth trying because it directly targets the two concrete flaws just diagnosed (fixed-band non-generalization, and no perspective handling) — but there's no guarantee it converges, especially on photos already shown to be at a genuine resolution floor (`images (1).jpg`). The quality-feedback half is lower-risk and independently useful regardless of how the CV tuning turns out.

## Approach

### 1. `gauge_reader.py` — adaptive tick-band search

Replace the fixed-band search in `_find_tick_mark_angles` with a scored search over candidate bands:
- Slide a band of fixed relative width (e.g. ~0.12×r) across candidate outer radii from ~0.35×r to ~1.0×r in small steps.
- For each candidate band, run the existing Hough-line + radial-orientation filtering (unchanged logic) to get a candidate tick-angle list.
- Score each candidate by (a) tick count (more is better, up to a sane cap) and (b) how *evenly spaced* the resulting ticks are — real tick marks have consistent angular gaps; a band that grabs the bezel edge or stray text won't. Compute this via the angular gaps from `_order_ticks_by_gap`'s existing gap logic, scoring on the coefficient of variation of consecutive gaps (lower = more self-consistent = more likely real).
- Pick the best-scoring band that clears a minimum tick count and a maximum gap-inconsistency threshold; return `[]` (safe decline, unchanged philosophy) if nothing clears the bar.

This directly targets the observed failure: a self-consistency score should reject the bezel-edge ring (which produced far fewer genuinely evenly-spaced radial hits than real minor+major tick marks would) in favor of the actual scale band, without needing to hardcode where that band is for any given gauge's proportions.

### 2. `gauge_reader.py` — ellipse rectification for off-axis photos

Add `_rectify_if_tilted(img_bgr, circle) -> tuple[np.ndarray, DialCircle]`:
- Within the existing circular ROI, find the dial's outer boundary contour and fit an ellipse (`cv2.fitEllipse`).
- If the ellipse's minor/major axis ratio is close to 1 (near-circular — most photos), skip rectification entirely (return the input unchanged) to avoid introducing warping artifacts where they're not needed.
- If notably elliptical (camera at an angle), compute the affine transform mapping that ellipse to a true circle of equivalent size, warp the ROI, and return the rectified image plus an updated `DialCircle` in the new coordinate space.
- Call this once at the top of `read_analog_gauge`, before needle/tick detection, so both benefit from the correction on tilted photos.

### 3. `image_quality.py` (new) — upload quality feedback

A small, gauge-independent module:
- `assess_blur(img_bgr) -> float`: Laplacian variance (standard sharpness metric) on the (possibly dial-cropped) region.
- `assess_gauge_quality(img_bgr, circle: DialCircle | None) -> str | None`: returns a human-readable warning string, or `None` if quality looks adequate. Flags: dial radius below an absolute pixel threshold (e.g. ~120px — informed by this session's observation that `images (1).jpg`'s ~92px-radius dial was where tick detection kept failing, while larger dials fared better), or blur score below a threshold. Only meaningful when a `DialCircle` was actually found — not applicable to plain text/digital images.

### 4. `main.py` — surface the warning

After the existing gauge-shaped-answer check, when a `DialCircle` is available (whether or not CV calibration ultimately succeeded), call `image_quality.assess_gauge_quality` and add an additive `"quality_warning": str | None` field to the JSON response. Never blocks the request or changes `text`/`verified`/`cv_override` — purely informational, so a calling app/UI can prompt "try a closer or sharper photo" on future uploads.

## Files to change
- `image_script/image_script/gauge_reader.py` — adaptive band search (replaces fixed `_TICK_BAND_INNER`/`_TICK_BAND_OUTER` constants with the scored search), ellipse rectification step.
- `image_script/image_script/image_quality.py` (new) — blur + dial-size quality assessment, independent of gauge geometry logic.
- `image_script/image_script/main.py` — wire in `quality_warning` field.
- `image_script/image_script/tests/test_gauge_reader.py` — re-verify existing cases still behave (safe-decline philosophy unchanged); add coverage for the new adaptive band search if it changes any currently-passing case's outcome.

## Verification
1. Re-run the debug visualizer (`python gauge_reader.py <image> --min --max`) against `images.jpg`, `images (1).jpg`, `images (3).jpg` — check whether the adaptive band now lands on the real printed scale on `images.jpg` (not the bezel), and whether ellipse rectification changes anything on photos with visible camera tilt.
2. Run `pytest` — existing safe-decline tests still pass.
3. Restart the server, re-test via curl against all sample images — confirm `quality_warning` appears (non-null) for the low-resolution close-up case and is `null` for adequately-sized/sharp photos, and that `text`/`verified`/`cv_override` behavior for non-gauge and digital cases is unaffected.
