"""End-to-end geometry check for tiled detection, using real crops and a fake model.

This is the test that would catch an off-by-one between a PIL crop rectangle and the
normalization applied to boxes coming back from it — the failure mode that puts every
detection slightly in the wrong place, which is easy to miss by eye and impossible to
catch with pure-arithmetic tests.

The "detector" is a perfect oracle: it finds the bright square in whatever crop it is
handed and reports it in crop-normalized coordinates, exactly as a real backend does.
So any drift between the true position and the merged output is the tiling code's.

Needs PIL (a hard project dependency) but no model, no numpy and no pydantic.

    python3 tests/test_tiling_integration.py      # or under pytest
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PIL import Image, ImageDraw
    HAVE_PIL = True
except ImportError:  # pragma: no cover - PIL is a project dependency
    HAVE_PIL = False

from utils.tiling import (  # noqa: E402
    TileDetection,
    is_truncated,
    map_tile_box_to_frame,
    merge_tile_detections,
    plan_tiles,
    select_tiles,
)

FRAME_W, FRAME_H = 2592, 1944
MODEL_W, MODEL_H = 640, 640
MIN_OBJECT_PX = 60


def make_frame(rects):
    """Black frame with white rectangles at the given pixel boxes."""
    image = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
    for (left, top, right, bottom) in rects:
        image.paste((255, 255, 255), (left, top, right, bottom))
    return image


def _components(mask, max_objects=64):
    """Bounding boxes of each connected bright region.

    A plain ``getbbox()`` would return the union of every bright pixel — one box
    spanning all objects — which no real detector does. Reporting per-object boxes
    matters here because the merge is specifically about deciding when two boxes are
    the same object, and a union-bbox oracle would never exercise that.

    Seeds are found from the first bright pixel on the top row of the remaining
    bounding box, so no full-image Python scan is needed; the flood fills themselves
    only touch object pixels.
    """
    work = mask.copy()
    boxes = []
    for _ in range(max_objects):
        bbox = work.getbbox()
        if bbox is None:
            break
        left, top, right, _bottom = bbox
        row = work.crop((left, top, right, top + 1)).tobytes()
        offset = next((i for i, value in enumerate(row) if value), None)
        if offset is None:  # pragma: no cover - getbbox guarantees a bright pixel
            break
        ImageDraw.floodfill(work, (left + offset, top), 128)
        component = work.point(lambda p: 255 if p == 128 else 0)
        boxes.append(component.getbbox())
        work = work.point(lambda p: 0 if p == 128 else p)
    return boxes


def oracle_detect(crop):
    """Report each bright region in crop-normalized coordinates."""
    mask = crop.convert("L").point(lambda p: 255 if p > 128 else 0)
    w, h = crop.size
    return [
        (left / w, top / h, right / w, bottom / h)
        for (left, top, right, bottom) in _components(mask)
    ]


def run_tiled(image, budget, now=0.0, tile_period=1.0):
    """Mirror of the endpoint's orchestration, using the oracle as the backend."""
    frame_w, frame_h = image.size
    plan = plan_tiles(frame_w, frame_h, MODEL_W, MODEL_H, MIN_OBJECT_PX)
    selected = select_tiles(plan, budget, tile_period, now)

    collected = []
    for tile in selected:
        crop = image if tile.full_frame else image.crop(tile.box)
        for box in oracle_detect(crop):
            collected.append(
                TileDetection(
                    label="object",
                    confidence=0.9,
                    box=map_tile_box_to_frame(box, tile, frame_w, frame_h),
                    tile_index=tile.index,
                    truncated=(
                        not tile.full_frame
                        and is_truncated(box, tile, frame_w, frame_h)
                    ),
                )
            )
    return merge_tile_detections(collected), plan, selected


def to_pixels(box):
    return (
        box[0] * FRAME_W,
        box[1] * FRAME_H,
        box[2] * FRAME_W,
        box[3] * FRAME_H,
    )


@unittest.skipUnless(HAVE_PIL, "PIL not installed")
class TestTiledGeometry(unittest.TestCase):
    def assert_box_close(self, got_norm, expected_px, tol=2.0):
        got = to_pixels(got_norm)
        for axis, (g, e) in enumerate(zip(got, expected_px)):
            self.assertLessEqual(
                abs(g - e), tol,
                f"axis {axis}: got {g:.1f}px, expected {e}px (tolerance {tol}px)\n"
                f"  full box got={tuple(round(v,1) for v in got)} expected={expected_px}",
            )

    def test_object_lands_at_its_true_frame_position(self):
        truth = (1400, 900, 1460, 960)
        image = make_frame([truth])
        merged, plan, _ = run_tiled(image, budget=len(plan_tiles(
            FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX)))
        self.assertGreaterEqual(len(merged), 1)
        self.assert_box_close(merged[0].box, truth)

    def test_position_is_correct_from_every_tile_that_sees_it(self):
        # The real risk: a tile-specific offset error that only shows up for crops
        # away from the origin. Check each covering tile independently.
        truth = (1400, 900, 1460, 960)
        image = make_frame([truth])
        plan = plan_tiles(FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX)

        covering = [
            t for t in plan
            if t.left <= truth[0] and t.top <= truth[1]
            and t.right >= truth[2] and t.bottom >= truth[3]
        ]
        self.assertGreater(len(covering), 1, "expected the object inside several tiles")

        for tile in covering:
            crop = image if tile.full_frame else image.crop(tile.box)
            boxes = oracle_detect(crop)
            self.assertEqual(len(boxes), 1, f"tile {tile.index} saw nothing")
            mapped = map_tile_box_to_frame(boxes[0], tile, FRAME_W, FRAME_H)
            self.assert_box_close(mapped, truth)

    def test_corner_objects_map_correctly(self):
        for truth in [
            (0, 0, 60, 60),                              # top-left
            (FRAME_W - 60, 0, FRAME_W, 60),              # top-right
            (0, FRAME_H - 60, 60, FRAME_H),              # bottom-left
            (FRAME_W - 60, FRAME_H - 60, FRAME_W, FRAME_H),  # bottom-right
        ]:
            with self.subTest(truth=truth):
                image = make_frame([truth])
                full_budget = len(plan_tiles(
                    FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX))
                merged, _, _ = run_tiled(image, budget=full_budget)
                self.assertGreaterEqual(len(merged), 1)
                self.assert_box_close(merged[0].box, truth)

    def test_object_on_a_tile_seam_is_not_duplicated(self):
        # An object straddling a seam is seen whole by the full-frame tile and
        # clipped by at least one crop. Exactly one box should survive the merge.
        plan = plan_tiles(FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX)
        seam = [t.right for t in plan if not t.full_frame and t.right < FRAME_W][0]
        truth = (seam - 40, 900, seam + 40, 980)
        image = make_frame([truth])

        merged, _, _ = run_tiled(image, budget=len(plan))
        self.assertEqual(
            len(merged), 1,
            f"expected one merged box, got {[to_pixels(m.box) for m in merged]}",
        )
        self.assert_box_close(merged[0].box, truth)

    def test_two_distinct_objects_both_reported(self):
        a = (300, 300, 360, 360)
        b = (2100, 1500, 2160, 1560)
        image = make_frame([a, b])
        full_budget = len(plan_tiles(FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX))
        merged, _, _ = run_tiled(image, budget=full_budget)
        self.assertEqual(
            len(merged), 2,
            f"both objects should survive: got {[to_pixels(m.box) for m in merged]}",
        )

        by_x = sorted(merged, key=lambda m: m.box[0])
        self.assert_box_close(by_x[0].box, a)
        self.assert_box_close(by_x[1].box, b)

    def test_empty_frame_yields_nothing(self):
        image = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
        merged, _, _ = run_tiled(image, budget=4)
        self.assertEqual(merged, [])

    def test_budget_is_respected_exactly(self):
        image = make_frame([(1400, 900, 1460, 960)])
        plan = plan_tiles(FRAME_W, FRAME_H, MODEL_W, MODEL_H, MIN_OBJECT_PX)
        for budget in (1, 2, 3):
            _, _, selected = run_tiled(image, budget=budget)
            self.assertEqual(len(selected), min(budget, len(plan)))

    def test_cost_is_independent_of_scene_content(self):
        # The invariant the whole design protects: an empty frame and a busy one
        # must issue exactly the same number of inferences.
        empty = Image.new("RGB", (FRAME_W, FRAME_H), (0, 0, 0))
        busy = make_frame([
            (x, y, x + 50, y + 50)
            for x in range(100, 2400, 400)
            for y in range(100, 1800, 400)
        ])
        _, _, sel_empty = run_tiled(empty, budget=3)
        _, _, sel_busy = run_tiled(busy, budget=3)
        self.assertEqual(len(sel_empty), len(sel_busy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
