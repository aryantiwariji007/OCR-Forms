# Pipeline Approach

## The actual flow

### Step 0 — Local, free, no API calls

- RapidOCR produces a rough text draft
- Classical CV finds the dial circle, then the needle angle (with the new guards: adaptive ink mask, tip-vs-counterweight by thickness, pivot-hub validation)

### Step 1 — Qwen/MaintServe, always exactly one call

- `verify_text`: decides *is this a gauge or plain text?* and gives a best-effort reading
- If it's plain text → **done, return it.** No further calls.

### Step 2 — Only if Qwen returned something gauge-shaped

First try the local digital decoder (`digital_display_reader`, free):

- Decodes 7-segment digits → that's the answer, `cv_override: true`, **done**
- This is what handles `gauge_digital_1` / `images (2).jpg`

If the digital decoder declines, go down the analog path:

- `read_scale_range` → **Claude first, Qwen as fallback**
- `read_endpoint_positions` → **Claude first, Qwen as fallback** (only if we have a needle angle)
- **We** compute the value ourselves: `interpolate_from_two_anchors(our needle angle, scale, endpoints)`
- Divergence check against Qwen's original guess — discard if too far apart
- If that fails, try tick-based `read_analog_gauge` (also divergence-checked)

### Step 3 — Claude last resort, only if no CV path produced a value

- Claude reads the whole image itself, its answer replaces Qwen's
- Ships with `quality_warning: "could not independently verify..."`
- This is what handles your digital 3/4/5

## The organizing principle

The split isn't really "CV, then Qwen, then Claude if unsure." It's **by task type**, based on what each is measurably good at:

| Task | Who does it | Why |
|---|---|---|
| Where is the needle pointing | **Always our own CV** — never a model | Both Qwen and Claude proved unreliable at fine spatial judgment. Claude read a counterweight as 3.7 bar when the truth was 0.35. |
| What numbers are printed on the scale, and where | **Claude** (Qwen fallback) | Literal transcription. Claude read a −1..5 compound scale correctly where Qwen said 0..5. |
| Is this a gauge or a document? Plain-text transcription | **Qwen** | Cheap, always runs, good enough at classification |
| Turning angle + scale into a number | **Plain arithmetic**, no model | Deterministic |

So: models read *text*, our geometry measures *position*, and arithmetic combines them. Claude is used pre-emptively for the reading tasks, not reactively when unsure.

## What it costs per image

| Image type | Calls |
|---|---|
| Plain text document | 1 Qwen |
| Digital that decodes locally | 1 Qwen |
| Analog where CV works | 1 Qwen + 2 Claude |
| Analog/digital where CV fails | 1 Qwen + 2-3 Claude |
