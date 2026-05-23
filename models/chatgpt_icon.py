#!/usr/bin/env python3
"""Generate a printable ChatGPT-style icon using CadQuery.

The model is intentionally parametric: edit the values in the PARAMETER AREA
below, then run this file again to regenerate the STL/STEP exports.
"""

from __future__ import annotations

import math
import struct
import sys
import zlib
from pathlib import Path


# =============================================================================
# PARAMETER AREA (millimetres)
# =============================================================================

# Base disk
BASE_DIAMETER = 80.0
BASE_THICKNESS = 3.0
BASE_EDGE_FILLET = 0.8

# Raised logo
LOGO_HEIGHT = 1.2
LOGO_Z_OFFSET = BASE_THICKNESS
# Tiny hidden overlap used only for the single-colour boolean union.  The
# exported separate logo remains exactly on top of the base for AMS alignment.
SINGLE_UNION_OVERLAP = 0.02

# Similarity-driven mode.  When enabled and the reference image exists, the top
# logo is vectorised from reference/chatgpt_reference.png into printable 2.5D
# polygon bands.  This keeps base/logo split for AMS while making the top-view
# silhouette closely match the supplied target image.
TRACE_REFERENCE_LOGO = True
REFERENCE_LOGO_PATH = "reference/chatgpt_reference.png"
TRACE_LOGO_DIAMETER = 56.0
TRACE_RESOLUTION = 288
TRACE_SIMPLIFY = 0.18
TRACE_THRESHOLD = 190
# CAD cleanup for bitmap trace mode.  The reference image is still sampled as a
# binary mask, but contiguous runs are unioned and then cleaned into CAD-friendly
# contours so exported logo edges do not keep the original pixel stair steps.
TRACE_CLOSE_BUFFER = 0.12
TRACE_OPEN_BUFFER = 0.045
TRACE_MIN_POLYGON_AREA = 0.35
TRACE_PREVIEW_SUPERSAMPLE = 3

# Six rounded folded bands.  Each lobe is made from short rounded capsules
# joined into a bent hook/chevron, then repeated every 60 degrees.  This is a
# printable ChatGPT-style interpretation, not an official curve trace.
PETAL_COUNT = 6
PETAL_WIDTH = 3.8
PETAL_START_ANGLE = 12.0
# Local XY path for one folded band before rotation.  Increase/decrease all
# coordinates together to scale the logo while preserving the interlocked look.
PETAL_PATH_POINTS = (
    (3.5, -7.8),
    (9.5, -17.0),
    (21.0, -16.5),
    (27.5, -5.0),
    (25.0, 6.0),
    (14.0, 11.0),
    (5.0, 5.5),
)

# Central negative space.  Hexagon is the default because it reads closer to the
# visible centre opening in the ChatGPT mark.
CENTER_VOID_SHAPE = "hexagon"
CENTER_VOID_DIAMETER = 10.0
CENTER_VOID_HEX_FLAT_DIAMETER = 8.0

# Printability guards
MIN_LINE_WIDTH = 0.8
MIN_GAP = 0.5
LOGO_OUTER_CLEARANCE_FROM_BASE_EDGE = 5.0

# Export names
EXPORT_DIR_NAME = "exports"
SINGLE_STL_NAME = "chatgpt_icon_single.stl"
STEP_NAME = "chatgpt_icon.step"
BASE_STL_NAME = "chatgpt_icon_base.stl"
LOGO_STL_NAME = "chatgpt_icon_logo.stl"
PREVIEW_PNG_NAME = "preview.png"

# =============================================================================


try:
    import cadquery as cq
    from cadquery import exporters
except Exception as exc:  # pragma: no cover - depends on local environment
    print(
        "ERROR: CadQuery is required but could not be imported.\n"
        "Install it with a CadQuery-capable environment, for example:\n"
        "  python -m pip install cadquery\n"
        f"Original import error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


try:  # Optional unless TRACE_REFERENCE_LOGO is enabled.
    from PIL import Image
    from shapely.geometry import MultiPolygon, Polygon, box
    from shapely.ops import unary_union
except Exception:  # pragma: no cover - validated at runtime when needed
    Image = None
    Polygon = None
    MultiPolygon = None
    box = None
    unary_union = None


def _project_root() -> Path:
    # Keep user-facing paths under /home/miku/projects even when that directory
    # is a symlink to another mount.
    return Path(__file__).absolute().parents[1]


def validate_parameters() -> None:
    """Fail early with clear messages for impossible or hard-to-print geometry."""
    errors: list[str] = []

    if BASE_DIAMETER <= 0:
        errors.append("BASE_DIAMETER must be greater than 0")
    if BASE_THICKNESS <= 0:
        errors.append("BASE_THICKNESS must be greater than 0")
    if LOGO_HEIGHT <= 0:
        errors.append("LOGO_HEIGHT must be greater than 0")
    if BASE_EDGE_FILLET < 0:
        errors.append("BASE_EDGE_FILLET cannot be negative")
    if BASE_EDGE_FILLET >= BASE_THICKNESS / 2:
        errors.append("BASE_EDGE_FILLET must be less than half BASE_THICKNESS")
    if PETAL_COUNT != 6:
        errors.append("PETAL_COUNT must stay at 6 for the requested icon style")
    if PETAL_WIDTH < MIN_LINE_WIDTH:
        errors.append(f"PETAL_WIDTH must be at least MIN_LINE_WIDTH ({MIN_LINE_WIDTH} mm)")
    if len(PETAL_PATH_POINTS) < 3:
        errors.append("PETAL_PATH_POINTS must contain at least three points for a folded band")
    for start, end in zip(PETAL_PATH_POINTS, PETAL_PATH_POINTS[1:]):
        segment_length = math.dist(start, end)
        if segment_length <= PETAL_WIDTH:
            errors.append(
                "Each PETAL_PATH_POINTS segment must be longer than PETAL_WIDTH: "
                f"segment {start}->{end} is {segment_length:.2f} mm"
            )
    if CENTER_VOID_DIAMETER < MIN_GAP:
        errors.append(f"CENTER_VOID_DIAMETER must be at least MIN_GAP ({MIN_GAP} mm)")
    if CENTER_VOID_SHAPE not in {"circle", "hexagon"}:
        errors.append('CENTER_VOID_SHAPE must be "circle" or "hexagon"')

    if TRACE_REFERENCE_LOGO:
        if Image is None or box is None or unary_union is None:
            errors.append("TRACE_REFERENCE_LOGO requires Pillow and Shapely")
        if TRACE_LOGO_DIAMETER <= 0:
            errors.append("TRACE_LOGO_DIAMETER must be greater than 0")
        if TRACE_LOGO_DIAMETER / 2 > BASE_DIAMETER / 2 - LOGO_OUTER_CLEARANCE_FROM_BASE_EDGE:
            errors.append("TRACE_LOGO_DIAMETER is too large for the base clearance")

    logo_outer_radius = max(math.hypot(x, y) for x, y in PETAL_PATH_POINTS) + PETAL_WIDTH / 2
    max_logo_radius = BASE_DIAMETER / 2 - LOGO_OUTER_CLEARANCE_FROM_BASE_EDGE
    if logo_outer_radius > max_logo_radius:
        errors.append(
            "Logo may extend too close to or beyond the base edge: "
            f"outer radius {logo_outer_radius:.2f} mm, allowed {max_logo_radius:.2f} mm"
        )

    centre_void_width = CENTER_VOID_HEX_FLAT_DIAMETER if CENTER_VOID_SHAPE == "hexagon" else CENTER_VOID_DIAMETER
    min_point_radius = min(math.hypot(x, y) for x, y in PETAL_PATH_POINTS)
    min_ring_width = min_point_radius - centre_void_width / 2 + PETAL_WIDTH / 2
    if min_ring_width < MIN_LINE_WIDTH:
        errors.append(
            "Central void is too large for the requested minimum line width: "
            f"available {min_ring_width:.2f} mm, required {MIN_LINE_WIDTH:.2f} mm"
        )

    if errors:
        raise ValueError("Parameter validation failed:\n- " + "\n- ".join(errors))


def make_base() -> cq.Workplane:
    """Create the round base disk with a small top/bottom edge fillet."""
    base = cq.Workplane("XY").circle(BASE_DIAMETER / 2).extrude(BASE_THICKNESS)
    if BASE_EDGE_FILLET > 0:
        # A cylinder's printable round-over edges are the circular top and bottom
        # perimeter edges.  ``|Z`` selects the cylinder seam edge, which CadQuery
        # cannot fillet as a solid edge here and raises "no suitable edges".
        base = base.edges("%CIRCLE").fillet(BASE_EDGE_FILLET)
    return base


def make_capsule_2d(length: float, width: float) -> cq.Workplane:
    """Create a 2D rounded capsule centred on the origin and aligned to X."""
    straight_length = length - width
    radius = width / 2
    left_x = -straight_length / 2
    right_x = straight_length / 2

    return (
        cq.Workplane("XY")
        .moveTo(left_x, radius)
        .lineTo(right_x, radius)
        .threePointArc((right_x + radius, 0), (right_x, -radius))
        .lineTo(left_x, -radius)
        .threePointArc((left_x - radius, 0), (left_x, radius))
        .close()
    )


def make_capsule_between_2d(start: tuple[float, float], end: tuple[float, float], width: float) -> cq.Workplane:
    """Create a 2D rounded capsule running between two local XY points."""
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 0:
        raise ValueError("Capsule segment length must be greater than zero")
    angle = math.degrees(math.atan2(dy, dx))
    midpoint = ((sx + ex) / 2, (sy + ey) / 2, 0)
    return make_capsule_2d(length, width).rotate((0, 0, 0), (0, 0, 1), angle).translate(midpoint)


def make_folded_band_solid() -> cq.Workplane:
    """Create one bent, rounded band solid from capsule segments."""
    band: cq.Workplane | None = None
    for start, end in zip(PETAL_PATH_POINTS, PETAL_PATH_POINTS[1:]):
        segment = make_capsule_between_2d(start, end, PETAL_WIDTH).extrude(LOGO_HEIGHT)
        band = segment if band is None else band.union(segment)
    if band is None:
        raise RuntimeError("No folded band segments were created")
    return band


def _reference_logo_mask() -> tuple[list[list[bool]], int, int]:
    """Load the reference mark as a fitted binary mask for trace mode."""
    if Image is None:
        raise RuntimeError("Pillow is required for TRACE_REFERENCE_LOGO")
    path = _project_root() / REFERENCE_LOGO_PATH
    if not path.is_file():
        raise FileNotFoundError(f"TRACE_REFERENCE_LOGO reference not found: {path}")

    image = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(bg, image).convert("L")
    mask = image.point(lambda px: 255 if px < TRACE_THRESHOLD else 0)
    bbox = mask.getbbox()
    if bbox is None:
        raise ValueError(f"Reference logo has no dark foreground pixels: {path}")
    cropped = mask.crop(bbox)
    cropped.thumbnail((TRACE_RESOLUTION, TRACE_RESOLUTION), Image.Resampling.LANCZOS)
    fitted = Image.new("L", (TRACE_RESOLUTION, TRACE_RESOLUTION), 0)
    fitted.paste(cropped, ((TRACE_RESOLUTION - cropped.width) // 2, (TRACE_RESOLUTION - cropped.height) // 2))
    data = []
    pixels = fitted.load()
    for y in range(fitted.height):
        data.append([pixels[x, y] >= 128 for x in range(fitted.width)])
    return data, fitted.width, fitted.height


def _polygon_to_workplane(poly: Polygon) -> cq.Workplane | None:  # type: ignore[valid-type]
    """Convert a Shapely polygon, including holes, into an extruded CadQuery solid."""
    if poly.is_empty or poly.area < 0.5:
        return None

    exterior = [(float(x), float(y)) for x, y in list(poly.exterior.coords)[:-1]]
    if len(exterior) < 3:
        return None
    workplane = cq.Workplane("XY").polyline(exterior).close()
    for interior in poly.interiors:
        hole = [(float(x), float(y)) for x, y in list(interior.coords)[:-1]]
        if len(hole) >= 3:
            workplane = workplane.polyline(hole).close()
    return workplane.extrude(LOGO_HEIGHT)


def make_traced_reference_logo() -> cq.Workplane:
    """Vectorise the supplied reference PNG into a printable raised logo."""
    if box is None or unary_union is None or Polygon is None or MultiPolygon is None:
        raise RuntimeError("Shapely is required for TRACE_REFERENCE_LOGO")
    mask, width, height = _reference_logo_mask()
    pixel = TRACE_LOGO_DIAMETER / max(width, height)
    min_x = -width * pixel / 2
    max_y = height * pixel / 2

    cells = []
    for y, row in enumerate(mask):
        for x, filled in enumerate(row):
            if not filled:
                continue
            x0 = min_x + x * pixel
            y1 = max_y - y * pixel
            cells.append(box(x0, y1 - pixel, x0 + pixel, y1))
    if not cells:
        raise ValueError("Reference trace produced no foreground cells")

    geometry = unary_union(cells).buffer(pixel * 0.45).buffer(-pixel * 0.45).simplify(TRACE_SIMPLIFY, preserve_topology=True)
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    logo: cq.Workplane | None = None
    for poly in polygons:
        solid = _polygon_to_workplane(poly)
        if solid is None:
            continue
        logo = solid if logo is None else logo.union(solid)
    if logo is None:
        raise RuntimeError("Reference trace produced no valid CadQuery solids")
    return logo.translate((0, 0, LOGO_Z_OFFSET))


def make_center_void() -> cq.Workplane:
    """Create the central negative-space cutter, tall enough to cut the logo."""
    cutter_height = LOGO_HEIGHT + 0.4
    z_start = LOGO_Z_OFFSET - 0.2

    if CENTER_VOID_SHAPE == "hexagon":
        radius = CENTER_VOID_HEX_FLAT_DIAMETER / math.sqrt(3)
        return (
            cq.Workplane("XY")
            .workplane(offset=z_start)
            .polygon(6, radius * 2)
            .extrude(cutter_height)
        )

    return (
        cq.Workplane("XY")
        .workplane(offset=z_start)
        .circle(CENTER_VOID_DIAMETER / 2)
        .extrude(cutter_height)
    )


def make_logo() -> cq.Workplane:
    """Create the six folded raised bands sitting directly on top of the base."""
    if TRACE_REFERENCE_LOGO:
        return make_traced_reference_logo()

    band_solid = make_folded_band_solid()
    logo: cq.Workplane | None = None

    for index in range(PETAL_COUNT):
        petal = (
            band_solid
            .rotate((0, 0, 0), (0, 0, 1), PETAL_START_ANGLE + index * 360.0 / PETAL_COUNT)
            .translate((0, 0, LOGO_Z_OFFSET))
        )
        logo = petal if logo is None else logo.union(petal)

    if logo is None:
        raise RuntimeError("No logo petals were created")

    return logo.cut(make_center_void())


def make_models() -> tuple[cq.Workplane, cq.Workplane, cq.Workplane, cq.Compound]:
    """Build base, separate logo, single-piece union, and reporting compound."""
    base = make_base()
    logo = make_logo()
    # For the one-piece STL, slightly sink a copy of the logo so the solids
    # overlap and CadQuery produces a true merged manifold instead of two solids
    # that merely touch at z=BASE_THICKNESS.  The top height stays effectively
    # the requested 4.2 mm and the separate AMS logo export is not shifted.
    single_logo = logo.translate((0, 0, -SINGLE_UNION_OVERLAP))
    single = base.union(single_logo)
    reporting_compound = cq.Compound.makeCompound([base.val(), logo.val()])
    return base, logo, single, reporting_compound


def export_models(base: cq.Workplane, logo: cq.Workplane, single: cq.Workplane) -> dict[str, Path]:
    """Export all requested files, preserving shared coordinates for multi-colour import."""
    export_dir = _project_root() / EXPORT_DIR_NAME
    export_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "single_stl": export_dir / SINGLE_STL_NAME,
        "step": export_dir / STEP_NAME,
        "base_stl": export_dir / BASE_STL_NAME,
        "logo_stl": export_dir / LOGO_STL_NAME,
        "preview_png": export_dir / PREVIEW_PNG_NAME,
    }

    exporters.export(single, str(paths["single_stl"]))

    assembly = cq.Assembly(name="chatgpt_icon")
    assembly.add(base, name="base")
    assembly.add(logo, name="logo")
    assembly.save(str(paths["step"]))

    exporters.export(base, str(paths["base_stl"]))
    exporters.export(logo, str(paths["logo_stl"]))
    export_preview_png(paths["preview_png"])
    return paths


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_rgba_png(path: Path, width: int, height: int, pixels: bytes) -> None:
    raw_rows = []
    stride = width * 4
    for y in range(height):
        raw_rows.append(b"\x00" + pixels[y * stride : (y + 1) * stride])
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(b"".join(raw_rows), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _inside_capsule(local_x: float, local_y: float, length: float, width: float) -> bool:
    radius = width / 2
    half_segment = (length - width) / 2
    qx = max(abs(local_x) - half_segment, 0.0)
    return qx * qx + local_y * local_y <= radius * radius


def _inside_capsule_between(x: float, y: float, start: tuple[float, float], end: tuple[float, float], width: float) -> bool:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared <= 0:
        return False
    t = max(0.0, min(1.0, ((x - sx) * dx + (y - sy) * dy) / length_squared))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    distance_x = x - closest_x
    distance_y = y - closest_y
    return distance_x * distance_x + distance_y * distance_y <= (width / 2) ** 2


def _inside_center_void(x: float, y: float) -> bool:
    if CENTER_VOID_SHAPE == "hexagon":
        # Pointy-top regular hexagon, sized by flat-to-flat diameter.
        flat_radius = CENTER_VOID_HEX_FLAT_DIAMETER / 2
        axial_x = abs(x)
        axial_y = abs(y)
        return axial_y <= flat_radius and math.sqrt(3) * axial_x + axial_y <= 2 * flat_radius
    return x * x + y * y <= (CENTER_VOID_DIAMETER / 2) ** 2


def export_preview_png(path: Path) -> None:
    """Write a dependency-free top-view PNG preview of the parametric footprint."""
    size = 900
    margin_px = 48
    scale = (size - 2 * margin_px) / BASE_DIAMETER
    half = size / 2
    base_radius = BASE_DIAMETER / 2

    bg = (250, 250, 247, 255)
    base_colour = (226, 226, 220, 255)
    edge_colour = (168, 168, 160, 255)
    logo_colour = (16, 163, 127, 255)
    pixels = bytearray(size * size * 4)

    traced_mask: list[list[bool]] | None = None
    traced_w = traced_h = 0
    if TRACE_REFERENCE_LOGO and (_project_root() / REFERENCE_LOGO_PATH).is_file():
        traced_mask, traced_w, traced_h = _reference_logo_mask()

    for py in range(size):
        y = (half - (py + 0.5)) / scale
        for px in range(size):
            x = ((px + 0.5) - half) / scale
            r2 = x * x + y * y
            colour = bg
            if r2 <= base_radius * base_radius:
                colour = base_colour
                if r2 >= (base_radius - 0.4) ** 2:
                    colour = edge_colour

            in_logo = False
            if traced_mask is not None:
                logo_scale = TRACE_LOGO_DIAMETER / max(traced_w, traced_h)
                tx = int((x + traced_w * logo_scale / 2) / logo_scale)
                ty = int((traced_h * logo_scale / 2 - y) / logo_scale)
                in_logo = 0 <= tx < traced_w and 0 <= ty < traced_h and traced_mask[ty][tx]
            else:
                for index in range(PETAL_COUNT):
                    orientation = math.radians(PETAL_START_ANGLE + index * 360.0 / PETAL_COUNT)
                    local_x = x * math.cos(orientation) + y * math.sin(orientation)
                    local_y = -x * math.sin(orientation) + y * math.cos(orientation)
                    for start, end in zip(PETAL_PATH_POINTS, PETAL_PATH_POINTS[1:]):
                        if _inside_capsule_between(local_x, local_y, start, end, PETAL_WIDTH):
                            in_logo = True
                            break
                    if in_logo:
                        break
                in_logo = in_logo and not _inside_center_void(x, y)
            if in_logo:
                colour = logo_colour

            offset = (py * size + px) * 4
            pixels[offset : offset + 4] = bytes(colour)

    _write_rgba_png(path, size, size, bytes(pixels))


def print_report(single: cq.Workplane, reporting_compound: cq.Compound, paths: dict[str, Path]) -> None:
    bbox = reporting_compound.BoundingBox()
    solid_count = len(reporting_compound.Solids())
    single_solid_count = len(single.val().Solids())

    print("ChatGPT-style icon generated successfully")
    print(
        "Bounding box (mm): "
        f"X={bbox.xlen:.3f}, Y={bbox.ylen:.3f}, Z={bbox.zlen:.3f}"
    )
    print(f"Maximum height (mm): {bbox.zmax:.3f}")
    print(f"Base diameter (mm): {BASE_DIAMETER:.3f}")
    print(f"Logo height (mm): {LOGO_HEIGHT:.3f}")
    print(f"Solid count (base + separate logo): {solid_count}")
    print(f"Single-colour merged solid count: {single_solid_count}")
    print("Export paths:")
    for label, path in paths.items():
        print(f"  {label}: {path}")


def main() -> int:
    try:
        validate_parameters()
        base, logo, single, reporting_compound = make_models()
        paths = export_models(base, logo, single)
        print_report(single, reporting_compound, paths)
        return 0
    except Exception as exc:
        print(f"ERROR: Failed to generate ChatGPT-style icon: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
