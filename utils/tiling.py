"""Fixed-budget tiled detection: geometry, tile selection and cross-tile merge.

The invariant this module exists to hold: **a request runs exactly T inferences**,
T being a caller-supplied constant. Scene content never changes T — it only changes
which T regions get looked at. Cost per request is therefore flat whether the frame
is empty or a tree is thrashing in the wind.

Everything here is a pure function over plain numbers. No PIL, no numpy, no pydantic,
no model — so tests/test_tiling.py exercises the whole module without a model file or
an inference runtime. Callers adapt to and from their own detection types at the edges.

Boxes are ``(x_min, y_min, x_max, y_max)`` normalized to whatever frame they belong to.
"""

from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, List, Sequence, Tuple

Box = Tuple[float, float, float, float]

# Smallest object, in *model input* pixels, that a YOLO-family grid reliably fires on.
# Used to convert a desired source-pixel object size into a required crop scale.
MIN_MODEL_PX = 24

# Hard ceiling on grid size. Sized against sweep time, not memory: at 1 Hz with T=4 a
# 24-tile grid sweeps in 8 s, inside a walking person's dwell time in frame. Much past
# ~30 the sweep outlasts the dwell and the coverage argument stops holding.
MAX_TILES = 32


@dataclass(frozen=True)
class Tile:
    """A crop rectangle in source-frame pixel coordinates.

    ``full_frame`` marks tile 0, which is always present and always selected, so
    large and near objects are still caught every cycle and behaviour never
    regresses below the untiled path.
    """

    index: int
    left: int
    top: int
    right: int
    bottom: int
    full_frame: bool = False

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def box(self) -> Tuple[int, int, int, int]:
        """Crop rect in PIL's ``(left, upper, right, lower)`` order."""
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class TileDetection:
    """A detection mapped into frame-normalized space, carrying its provenance.

    ``truncated`` means the box touches a tile edge that is *not* also a frame edge,
    so the object is probably clipped and this copy is partial. The merge prefers a
    non-truncated copy of the same object from a neighbouring tile.
    """

    label: str
    confidence: float
    box: Box
    tile_index: int
    truncated: bool = False


def _region_dims(img_w: int, img_h: int, model_w: int, model_h: int,
                 required_scale: float) -> Tuple[int, int]:
    """Crop size that holds a ``min_object_px`` object at MIN_MODEL_PX, aspect-matched.

    **The crop must have the model's aspect ratio.** The backend letterboxes each crop
    independently, so it scales by ``min(model_w/region_w, model_h/region_h)`` and pads
    the slack axis with grey bars. A crop shaped unlike the model input therefore wastes
    part of the input on padding *and* is scaled by whichever axis is relatively larger.

    Sizing the axes independently — the obvious reading of "region = model / scale" —
    gets this wrong whenever one axis clamps to the frame and the other does not. On
    1080p at min_object_px=60 it yields a 1600x1080 crop against a 640x640 input: 32% of
    the input is padding and the effective scale is 0.400, when a square 1080x1080 crop
    of the same frame would give 0.593 for the same tile count.

    So parameterise the crop as ``(model_w * t, model_h * t)`` and take the largest ``t``
    that still fits the frame and still meets the scale requirement::

        t = min(img_w/model_w, img_h/model_h, 1/required_scale)

    The third term is the accuracy floor; the first two are the frame. Whichever binds,
    the crop keeps the model's shape, so the letterbox never pads.

    Rounds **down**, so ``region <= model_dim * t`` and the letterbox scale stays at or
    above ``required_scale``. Rounding up would land the object fractionally under
    MIN_MODEL_PX — the opposite of the guarantee this module advertises.

    Shared by plan_tiles and tile_grid_overflowed so the two cannot disagree about how
    big a tile is, and therefore cannot disagree about whether the grid overflowed.
    """
    t = min(img_w / float(model_w), img_h / float(model_h), 1.0 / required_scale)
    return max(1, int(floor(model_w * t))), max(1, int(floor(model_h * t)))


def _axis_starts(total: int, region: int, stride: int) -> List[int]:
    """Crop origins along one axis: first at 0, last flush to the far edge, evenly spread.

    Every crop is the same size, which matters because the backend letterboxes each one
    independently — a short final crop would be scaled differently from its neighbours.

    Spacing them **evenly** rather than walking by ``stride`` and flushing the last one
    is what keeps the axis gapless by construction. Walking-and-flushing puts the final
    crop wherever the remainder falls, which can land it either almost on top of its
    predecessor (a wasted inference that dilutes the rotation pool) or too far past it
    (an uncovered strip). Distributing the span over a whole number of steps cannot do
    either: the spacing is uniform, and capping it at ``region`` guarantees consecutive
    crops touch.

    The chosen spacing is at most the requested ``stride``, so the real overlap is always
    at least what the caller asked for.
    """
    if region >= total:
        return [0]
    stride = max(1, stride)
    span = total - region

    # A second crop that breaks little new ground is not worth an inference. Measured
    # against the region, not the stride: what matters is the fraction of the crop that
    # is new, and a stride-relative test lets through pairs overlapping 90%+ (4K at
    # min_object_px=74 wanted crops at y=0 and y=187 of a 1973px region).
    # The strip left uncovered is at most this fraction of the axis, and tile 0 — the
    # full frame — still inspects it every cycle. It sits at the far edge, which for a
    # fixed camera is foreground, where objects are large and magnification is moot.
    if span < region * 0.15:
        return [0]

    # Number of gaps between starts. The stride term sets the overlap; the region term
    # guarantees no gap even at overlap=0. The small tolerance stops a frame that
    # overshoots the stride by a hair from buying a whole extra crop for it — 1920 wide
    # with a 1080 region overshoots by 4%, and paying for a third column there would
    # give 61% overlap where 25% was asked for.
    steps = max(
        1,
        int(ceil(span / float(region))),
        int(ceil(span / float(stride) - 0.05)),
    )
    return [int(round(i * span / float(steps))) for i in range(steps + 1)]


def plan_tiles(
    img_w: int,
    img_h: int,
    model_w: int,
    model_h: int,
    min_object_px: int,
    overlap: float = 0.25,
    max_tiles: int = MAX_TILES,
) -> List[Tile]:
    """Build the tile grid for one frame.

    ``min_object_px`` is the smallest object, in *source* pixels, that should still
    be resolvable. The crop size follows from it::

        required_scale = MIN_MODEL_PX / min_object_px    # 24/60 = 0.40
        region         = model_input / required_scale    # 640/0.40 = 1600 px
        stride         = region * (1 - overlap)          # 25% overlap -> 1200 px

    Tile 0 is always the full frame. When the computed region already covers the
    frame, the grid collapses to that single tile rather than paying twice for the
    same pixels.
    """
    full = Tile(index=0, left=0, top=0, right=img_w, bottom=img_h, full_frame=True)

    if min_object_px is None or min_object_px <= 0:
        return [full]
    if img_w <= 0 or img_h <= 0 or model_w <= 0 or model_h <= 0:
        return [full]
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if max_tiles < 1:
        return [full]

    required_scale = MIN_MODEL_PX / float(min_object_px)
    if required_scale <= 0:
        return [full]

    # Degenerate: the full frame already resolves the target, so cropping would buy
    # nothing. Tested on scale rather than geometry — with aspect-matched crops a
    # region can be smaller than the frame on both axes yet still offer no gain.
    if min(model_w / float(img_w), model_h / float(img_h)) >= required_scale:
        return [full]

    region_w, region_h = _region_dims(img_w, img_h, model_w, model_h, required_scale)
    if region_w >= img_w and region_h >= img_h:
        return [full]

    stride_w = int(region_w * (1.0 - overlap))
    stride_h = int(region_h * (1.0 - overlap))

    tiles = [full]
    for top in _axis_starts(img_h, region_h, stride_h):
        for left in _axis_starts(img_w, region_w, stride_w):
            tiles.append(
                Tile(
                    index=len(tiles),
                    left=left,
                    top=top,
                    right=left + region_w,
                    bottom=top + region_h,
                )
            )

    if len(tiles) > max_tiles:
        # Keep tile 0 plus the first (max_tiles - 1) crops. Callers should treat this
        # as a misconfiguration signal: min_object_px is set finer than the frame can
        # sweep in reasonable time.
        tiles = tiles[:max_tiles]

    return tiles


def tile_grid_overflowed(
    img_w: int,
    img_h: int,
    model_w: int,
    model_h: int,
    min_object_px: int,
    overlap: float = 0.25,
    max_tiles: int = MAX_TILES,
) -> bool:
    """True when plan_tiles had to truncate the grid — worth logging a warning on."""
    if min_object_px is None or min_object_px <= 0:
        return False
    if img_w <= 0 or img_h <= 0 or model_w <= 0 or model_h <= 0:
        return False
    required_scale = MIN_MODEL_PX / float(min_object_px)
    # Mirrors plan_tiles exactly, including the scale-based degenerate check.
    if min(model_w / float(img_w), model_h / float(img_h)) >= required_scale:
        return False
    region_w, region_h = _region_dims(img_w, img_h, model_w, model_h, required_scale)
    if region_w >= img_w and region_h >= img_h:
        return False
    stride_w = int(region_w * (1.0 - overlap))
    stride_h = int(region_h * (1.0 - overlap))
    count = 1 + len(_axis_starts(img_h, region_h, stride_h)) * len(
        _axis_starts(img_w, region_w, stride_w)
    )
    return count > max_tiles


def select_tiles(
    plan: Sequence[Tile],
    budget: int,
    tile_period: float,
    now: float,
) -> List[Tile]:
    """Pick exactly ``budget`` tiles: tile 0 plus a time-rotated slice of the rest.

    The cursor is derived from the clock, not from stored state — no per-stream
    server state, no eviction, no multi-worker coordination, nothing to corrupt
    across restarts. Two workers handling successive frames agree by construction.

    **``tile_period`` must match how often the caller actually fires.** This is a
    contract, not a hint. The cursor advances linearly with the clock, so a caller
    whose period is an exact multiple of ``tile_period`` samples the same residues
    forever and the tiles in between are never visited at all. Concretely, with a
    4-tile pool and T=3, a caller firing every 2 s against ``tile_period=1`` selects
    tiles 1 and 2 on every single cycle; tiles 3 and 4 starve permanently. Passing
    ``tile_period=2`` restores full coverage.

    The rate to match is the *effective* one, which on lightNVR's keyframe-gated
    path is the GOP length rather than the configured ``detection_interval`` — those
    differ whenever the interval is shorter than the keyframe spacing.

    A coprime stride would absorb the mismatch, but it stretches the nominal sweep of
    a 24-tile grid from 8 cycles to roughly 12, which is the wrong trade against a
    misconfiguration that ``tile_period`` already fixes.

    Minor drift is harmless: a tile is occasionally visited twice and another skipped
    within a sweep, but coverage stays uniform over time. Only exact commensurability
    starves.
    """
    if not plan:
        return []
    head = plan[0]
    pool = list(plan[1:])
    if budget <= 1 or not pool:
        return [head]

    n_rot = min(budget - 1, len(pool))
    period = tile_period if tile_period and tile_period > 0 else 1.0
    step = int(now // period)
    cursor = (step * n_rot) % len(pool)

    selected = [pool[(cursor + i) % len(pool)] for i in range(n_rot)]
    return [head] + selected


def sweep_cycles(plan_size: int, budget: int) -> int:
    """Cycles needed to visit every tile once. ``ceil(K / (T-1))`` with K the pool."""
    pool = max(0, plan_size - 1)
    if pool == 0:
        return 1
    n_rot = max(1, min(budget - 1, pool))
    return int(ceil(pool / float(n_rot)))


def map_tile_box_to_frame(
    box: Box,
    tile: Tile,
    frame_w: int,
    frame_h: int,
) -> Box:
    """Convert a tile-normalized box to frame-normalized coordinates."""
    if frame_w <= 0 or frame_h <= 0:
        return (0.0, 0.0, 0.0, 0.0)

    x_min, y_min, x_max, y_max = box
    fx_min = (tile.left + x_min * tile.width) / frame_w
    fx_max = (tile.left + x_max * tile.width) / frame_w
    fy_min = (tile.top + y_min * tile.height) / frame_h
    fy_max = (tile.top + y_max * tile.height) / frame_h

    return (
        _clamp01(fx_min),
        _clamp01(fy_min),
        _clamp01(fx_max),
        _clamp01(fy_max),
    )


def is_truncated(
    box: Box,
    tile: Tile,
    frame_w: int,
    frame_h: int,
    edge_epsilon: float = 0.01,
) -> bool:
    """True when a tile-normalized box touches a tile edge that isn't a frame edge.

    Such a box is probably a clipped fragment of an object that continues into the
    neighbouring tile, so its extent and confidence are both understated. A box
    touching a *frame* edge is not truncated — nothing lies beyond it to recover.
    """
    x_min, y_min, x_max, y_max = box

    if x_min <= edge_epsilon and tile.left > 0:
        return True
    if y_min <= edge_epsilon and tile.top > 0:
        return True
    if x_max >= 1.0 - edge_epsilon and tile.right < frame_w:
        return True
    if y_max >= 1.0 - edge_epsilon and tile.bottom < frame_h:
        return True
    return False


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def box_area(box: Box) -> float:
    x_min, y_min, x_max, y_max = box
    return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


def box_iou(a: Box, b: Box) -> float:
    """Intersection over union of two normalized boxes."""
    inter = _intersection_area(a, b)
    if inter <= 0.0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0.0 else 0.0


def containment(inner: Box, outer: Box) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    area = box_area(inner)
    if area <= 0.0:
        return 0.0
    return _intersection_area(inner, outer) / area


def _intersection_area(a: Box, b: Box) -> float:
    x_min = max(a[0], b[0])
    y_min = max(a[1], b[1])
    x_max = min(a[2], b[2])
    y_max = min(a[3], b[3])
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)


def merge_tile_detections(
    detections: Iterable[TileDetection],
    iou_threshold: float = 0.45,
    containment_threshold: float = 0.8,
) -> List[TileDetection]:
    """Class-aware NMS across tiles, biased against edge-truncated copies.

    Ordering is ``(truncated, -confidence)``: every intact copy is considered before
    any clipped one, so when the same object is seen whole in one tile and clipped in
    another, the whole copy is kept and the fragment suppressed. Within a truncation
    class, ordinary confidence ordering applies.

    Plain IoU is not enough on its own — a small fragment of a large object can fall
    below the IoU threshold against the full box and survive as a phantom duplicate.
    The containment test catches that case: a truncated box mostly inside an already
    kept box of the same label is dropped regardless of IoU.
    """
    ordered = sorted(detections, key=lambda d: (d.truncated, -d.confidence))

    kept: List[TileDetection] = []
    for candidate in ordered:
        suppressed = False
        for winner in kept:
            if winner.label != candidate.label:
                continue
            if box_iou(winner.box, candidate.box) > iou_threshold:
                suppressed = True
                break
            if (
                candidate.truncated
                and containment(candidate.box, winner.box) >= containment_threshold
            ):
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    kept.sort(key=lambda d: -d.confidence)
    return kept
