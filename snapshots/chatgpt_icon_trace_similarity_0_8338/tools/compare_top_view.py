#!/usr/bin/env python3
"""Compare generated top-view preview against a reference ChatGPT mark image.

This is a lightweight visual gate for the CAD iteration loop.  It intentionally
uses Pillow for robust image loading/resizing, then computes shape-oriented
metrics on binary silhouettes so colour differences do not dominate the score.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_CANDIDATE = PROJECT_ROOT / "exports" / "preview.png"
DEFAULT_REFERENCE = PROJECT_ROOT / "reference" / "chatgpt_reference.png"
DEFAULT_REPORT = PROJECT_ROOT / "exports" / "similarity_report.json"
DEFAULT_SIZE = 512
DEFAULT_TARGET_SCORE = 0.72


def _load_pillow():
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageOps
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Pillow is required for image comparison. Install with: python -m pip install pillow"
        ) from exc
    return Image, ImageChops, ImageDraw, ImageOps


def _fit_to_square(mask, size: int, image_modules: tuple[Any, Any, Any, Any]):
    Image, _ImageChops, ImageDraw, _ImageOps = image_modules
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", (size, size), 0)

    cropped = mask.crop(bbox)
    scale = min((size * 0.82) / cropped.width, (size * 0.82) / cropped.height)
    new_size = (max(1, round(cropped.width * scale)), max(1, round(cropped.height * scale)))
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    normalised = Image.new("L", (size, size), 0)
    normalised.paste(cropped, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return normalised.point(lambda px: 255 if px >= 128 else 0)


def _candidate_logo_mask(path: Path, size: int, image_modules: tuple[Any, Any, Any, Any]):
    Image, _ImageChops, _ImageDraw, _ImageOps = image_modules
    image = Image.open(path).convert("RGBA")
    # The generated preview uses GPT-green logo over a pale base.  This selects
    # saturated/dark foreground strokes and ignores the circular base.
    mask = Image.new("L", image.size, 0)
    pixels = image.load()
    out = mask.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            if a > 10 and ((g > r + 35 and g > b + 20) or (r + g + b < 180)):
                out[x, y] = 255
    return _fit_to_square(mask, size, image_modules)


def _reference_logo_mask(path: Path, size: int, image_modules: tuple[Any, Any, Any, Any]):
    Image, _ImageChops, _ImageDraw, ImageOps = image_modules
    image = Image.open(path).convert("RGBA")
    mask = Image.new("L", image.size, 0)
    pixels = image.load()
    out = mask.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b, a = pixels[x, y]
            # Accept typical references: black logo on white/transparent, or the
            # generated green preview when used as a self-test reference.
            if a > 10 and ((r + g + b < 300 and max(r, g, b) - min(r, g, b) < 55) or (g > r + 35 and g > b + 20)):
                out[x, y] = 255
    return _fit_to_square(mask, size, image_modules)


def _count(mask) -> int:
    hist = mask.histogram()
    return sum(hist[128:])


def _intersection_union(a, b, image_modules: tuple[Any, Any, Any, Any]) -> tuple[int, int]:
    _Image, ImageChops, _ImageDraw, _ImageOps = image_modules
    inter = ImageChops.multiply(a, b)
    union = ImageChops.lighter(a, b)
    return _count(inter), _count(union)


def _edge_mask(mask, image_modules: tuple[Any, Any, Any, Any]):
    _Image, ImageChops, _ImageDraw, _ImageOps = image_modules
    # Morphological edge: mask - eroded(mask) using min filter.
    from PIL import ImageFilter

    eroded = mask.filter(ImageFilter.MinFilter(5))
    return ImageChops.subtract(mask, eroded)


def compare(candidate: Path, reference: Path, size: int = DEFAULT_SIZE, target: float = DEFAULT_TARGET_SCORE) -> dict[str, Any]:
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate preview not found: {candidate}")
    if not reference.is_file():
        raise FileNotFoundError(
            f"Reference image not found: {reference}. Save the target logo there or pass --reference."
        )

    image_modules = _load_pillow()
    candidate_mask = _candidate_logo_mask(candidate, size, image_modules)
    reference_mask = _reference_logo_mask(reference, size, image_modules)

    inter, union = _intersection_union(candidate_mask, reference_mask, image_modules)
    iou = inter / union if union else 0.0

    candidate_area = _count(candidate_mask)
    reference_area = _count(reference_mask)
    area_ratio = min(candidate_area, reference_area) / max(candidate_area, reference_area) if max(candidate_area, reference_area) else 0.0

    cand_edge = _edge_mask(candidate_mask, image_modules)
    ref_edge = _edge_mask(reference_mask, image_modules)
    edge_inter, edge_union = _intersection_union(cand_edge, ref_edge, image_modules)
    edge_iou = edge_inter / edge_union if edge_union else 0.0

    # Weighted for rough CAD iteration: IoU dominates, area/edge keep scale and
    # silhouette detail honest.  "Basic close" is intentionally lower than exact
    # logo matching because this model is a printable approximation.
    score = 0.62 * iou + 0.23 * area_ratio + 0.15 * edge_iou
    return {
        "ok": score >= target,
        "score": round(score, 4),
        "target": target,
        "iou": round(iou, 4),
        "area_ratio": round(area_ratio, 4),
        "edge_iou": round(edge_iou, 4),
        "candidate_area_px": candidate_area,
        "reference_area_px": reference_area,
        "candidate": str(candidate),
        "reference": str(reference),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generated top-view preview to a reference logo image.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE, help=f"Generated preview PNG (default: {DEFAULT_CANDIDATE})")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help=f"Reference PNG/JPG (default: {DEFAULT_REFERENCE})")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help=f"JSON report output (default: {DEFAULT_REPORT})")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"Normalised comparison size (default: {DEFAULT_SIZE})")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET_SCORE, help=f"Pass threshold (default: {DEFAULT_TARGET_SCORE})")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = compare(args.candidate, args.reference, args.size, args.target)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 2
    except Exception as exc:
        result = {"ok": False, "error": str(exc), "candidate": str(args.candidate), "reference": str(args.reference)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
