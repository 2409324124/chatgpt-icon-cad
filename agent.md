# Agent Notes: Successful CAD and AMS Workflow

This project produced a successful printable ChatGPT-style icon by treating CAD generation, visual matching, and slicer import as one loop. Use this file as the agent playbook for future icon or badge projects.

## Goal

Generate a small 3D-printable icon with:

- A snug round base.
- A raised logo that can be printed in a second AMS colour.
- A single-colour fallback STL.
- A browser preview and top-view similarity score.
- Export files that work directly in Bambu Studio.

## File contract

Keep these outputs stable:

```text
exports/chatgpt_icon_base.stl     # AMS base part
exports/chatgpt_icon_logo.stl     # AMS logo part, same coordinates as base
exports/chatgpt_icon_single.stl   # single-colour merged model
exports/chatgpt_icon.step         # CAD assembly with base/logo
exports/preview.png               # top-view preview for similarity scoring
```

For Bambu Studio, import `chatgpt_icon_base.stl` and `chatgpt_icon_logo.stl` together and choose the “single object with multiple parts” option. Do not move the two parts independently. They are already aligned.

## What worked

### 1. Split the model for AMS, but also export a merged fallback

Do not rely on slicer painting for clean multi-colour logos. Export the base and logo as two separate STL files in the same coordinate system:

```python
exporters.export(base, "exports/chatgpt_icon_base.stl")
exporters.export(logo, "exports/chatgpt_icon_logo.stl")
```

Also export a single-colour version by sinking the logo slightly before boolean union:

```python
single_logo = logo.translate((0, 0, -SINGLE_UNION_OVERLAP))
single = base.union(single_logo)
```

The hidden overlap prevents CadQuery from leaving two solids that only touch at a face.

### 2. Build the logo first, then size the base from the real logo bounding box

The first version used a fixed 80 mm base, which looked like a large disk around a small logo. The successful version computes the base dynamically:

```python
logo = make_logo()
logo_bb = logo.val().BoundingBox()
logo_max_dim = max(logo_bb.xlen, logo_bb.ylen)
base_diameter = logo_max_dim + 2 * BASE_MARGIN_PER_SIDE
base = make_base(base_diameter)
```

The current good setting is:

```python
BASE_MARGIN_PER_SIDE = 1.5
```

That makes the base about 3 mm larger than the logo overall, so the rim is visible without dominating the design.

### 3. Keep the Z coordinates slicer-friendly

The base occupies `z = 0..BASE_THICKNESS`. The separate logo starts at `LOGO_Z_OFFSET = BASE_THICKNESS`.

```python
LOGO_Z_OFFSET = BASE_THICKNESS
```

This makes the two STL files align in Bambu Studio without manual lifting or repositioning.

### 4. Use reference-image tracing when visual fidelity matters

The parametric folded-band logo is useful, but the best match came from tracing `reference/chatgpt_reference.png` into CAD-friendly polygons:

```python
TRACE_REFERENCE_LOGO = True
TRACE_CLOSE_BUFFER = 0.18
TRACE_OPEN_BUFFER = 0.07
TRACE_SIMPLIFY = 0.12
```

The close/open cleanup matters. It removes pixel stair steps and produces smoother, printable edges.

### 5. Score shape and vector quality separately

The similarity check should not only compare pixels. It should also penalize jagged edges. The successful scoring split is:

```text
Shape:          82%  silhouette IoU, area ratio, edge IoU
Vector quality: 18%  straightness_score, curvature_score
```

Definitions used here:

- `straightness_score`: perimeter stability after morphological smoothing.
- `curvature_score`: shape IoU after morphological smoothing.

This caught the difference between “looks roughly right” and “has clean CAD edges.”

## Verification gates

Run these before saying the model is done:

```bash
python3 -m py_compile models/chatgpt_icon.py tools/compare_top_view.py tools/search_logo_params.py tools/serve_preview.py
python3 models/chatgpt_icon.py
python3 tools/compare_top_view.py --target 0.72
```

Expected successful output should include:

```text
Solid count (base + separate logo): 2
Single-colour merged solid count: 1
Base-logo clearance: about 3.000 mm (1.5 per side)
Similarity score: >= 0.72
```

The successful tight-base version produced approximately:

```text
Base diameter: 35.812 mm
Logo bounding box: X=32.812 mm, Y=32.666 mm
Similarity score: 0.9131
```

## Review checklist for future agents

Before changing the model, check these points:

- Does `chatgpt_icon_base.stl` contain only the base?
- Does `chatgpt_icon_logo.stl` contain only the raised logo and keep its original Z position?
- Does `chatgpt_icon_single.stl` report one merged solid?
- Does `chatgpt_icon.step` contain named `base` and `logo` assembly parts?
- Is the base diameter computed from the actual logo bounding box, not hardcoded?
- Is the base margin small enough to look intentional, usually 1 to 2 mm per side?
- Does `preview.png` use the actual computed base diameter?
- Does `print_report()` use `logo.val().BoundingBox()` for the logo bbox, not `reporting_compound.Solids()[1]`?
- Does `tools/compare_top_view.py --target 0.72` pass?
- If this is for AMS, did you test by importing base and logo STLs together?

## Common failure modes

### Base looks too large

Do not tune a fixed `BASE_DIAMETER`. Keep the dynamic formula and adjust:

```python
BASE_MARGIN_PER_SIDE
```

### Bambu Studio imports the parts as separate objects

Select both STL files at the same time before importing. If prompted, choose the option to load them as one object with multiple parts.

### Single STL is not actually one solid

Increase `SINGLE_UNION_OVERLAP` slightly, then regenerate. Keep it tiny so the visible height does not change.

### Similarity is high but edges look bad

Inspect `straightness_score` and `curvature_score`, not just the total score. Tune trace cleanup buffers before changing the logo scale.

### Reported logo size is wrong

Use the real logo workplane for measurement:

```python
logo.val().BoundingBox()
```

Do not infer the logo from `reporting_compound.Solids()[1]`, because a cut can split the logo into multiple solids.

## Good agent workflow

1. Freeze the acceptance criteria: AMS two-part import, single merged fallback, similarity target, printability limits.
2. Make the smallest CAD change.
3. Regenerate exports.
4. Run syntax, model generation, and similarity gates.
5. Review the generated report numbers, not just file existence.
6. Open the browser preview for a human visual check.
7. Only commit after the exported files and verification report match the new design.

The key lesson: CAD success came from closing the loop between geometry, slicer behaviour, and measured visual similarity. Treat exported STL/STEP files as product artifacts, not incidental build output.
