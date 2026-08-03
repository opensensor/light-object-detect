"""Tests for utils/tiling.py — tile geometry, rotation and cross-tile merge.

Stdlib only, matching test_onnx_providers.py: utils/tiling.py deliberately has no
third-party imports, so this runs on a machine with no numpy, pydantic or model file.

    python3 tests/test_tiling.py      # or: pytest tests/test_tiling.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tiling import (  # noqa: E402
    MAX_TILES,
    MIN_MODEL_PX,
    Tile,
    TileDetection,
    box_iou,
    containment,
    is_truncated,
    map_tile_box_to_frame,
    merge_tile_detections,
    plan_tiles,
    select_tiles,
    sweep_cycles,
    tile_grid_overflowed,
)


class TestPlanTiles(unittest.TestCase):
    def test_tile_zero_is_always_the_full_frame(self):
        for w, h in [(1920, 1080), (2592, 1944), (640, 640), (4000, 600)]:
            plan = plan_tiles(w, h, 640, 640, 60)
            self.assertTrue(plan[0].full_frame)
            self.assertEqual(plan[0].box, (0, 0, w, h))

    def test_1080p_grid_covers_the_frame(self):
        plan = plan_tiles(1920, 1080, 640, 640, 60)
        self.assertGreater(len(plan), 1)
        # Every column of the frame must fall inside at least one non-full tile.
        crops = [t for t in plan if not t.full_frame]
        for x in range(0, 1920, 37):
            self.assertTrue(
                any(t.left <= x < t.right for t in crops),
                f"column {x} uncovered",
            )

    def test_5mp_grid_covers_both_axes(self):
        plan = plan_tiles(2592, 1944, 640, 640, 60)
        crops = [t for t in plan if not t.full_frame]
        self.assertGreater(len(crops), 1)
        for y in range(0, 1944, 41):
            self.assertTrue(any(t.top <= y < t.bottom for t in crops), f"row {y}")
        for x in range(0, 2592, 41):
            self.assertTrue(any(t.left <= x < t.right for t in crops), f"col {x}")

    def test_crops_are_uniformly_sized(self):
        # The backend letterboxes each crop independently, so a short edge tile would
        # be scaled differently from its neighbours. Last row/col flush to the edge.
        plan = plan_tiles(2592, 1944, 640, 640, 60)
        crops = [t for t in plan if not t.full_frame]
        widths = {t.width for t in crops}
        heights = {t.height for t in crops}
        self.assertEqual(len(widths), 1, f"ragged widths: {widths}")
        self.assertEqual(len(heights), 1, f"ragged heights: {heights}")

    def test_crops_stay_inside_the_frame(self):
        plan = plan_tiles(1920, 1080, 640, 640, 45)
        for t in plan:
            self.assertGreaterEqual(t.left, 0)
            self.assertGreaterEqual(t.top, 0)
            self.assertLessEqual(t.right, 1920)
            self.assertLessEqual(t.bottom, 1080)

    def test_degenerate_case_collapses_to_full_frame(self):
        # One crop already covers the frame: tiling would re-inspect the same pixels
        # at the same scale, so the grid must collapse rather than pay twice.
        self.assertEqual(len(plan_tiles(640, 480, 640, 640, 60)), 1)
        self.assertEqual(len(plan_tiles(1500, 1200, 640, 640, 60)), 1)

    def test_native_scale_request_still_tiles(self):
        # min_object_px == MIN_MODEL_PX asks for native scale: a 640 crop out of a
        # larger frame is a real gain, so this must NOT collapse.
        plan = plan_tiles(1000, 800, 640, 640, MIN_MODEL_PX)
        self.assertGreater(len(plan), 1)
        crop = [t for t in plan if not t.full_frame][0]
        self.assertEqual((crop.width, crop.height), (640, 640))

    def test_required_scale_actually_resolves_the_target_object(self):
        # The point of the whole exercise: a min_object_px object must survive the
        # crop -> model-input scaling at or above MIN_MODEL_PX.
        min_object_px = 60
        plan = plan_tiles(2592, 1944, 640, 640, min_object_px)
        crop = [t for t in plan if not t.full_frame][0]
        scale = min(640 / crop.width, 640 / crop.height)
        self.assertGreaterEqual(min_object_px * scale, MIN_MODEL_PX - 1e-6)

    def test_extreme_aspect_ratio(self):
        plan = plan_tiles(4000, 400, 640, 640, 60)
        crops = [t for t in plan if not t.full_frame]
        for x in range(0, 4000, 53):
            self.assertTrue(any(t.left <= x < t.right for t in crops), f"col {x}")

    def test_grid_is_capped_and_overflow_is_reported(self):
        # A very small min_object_px explodes the grid; the cap holds and the
        # companion predicate tells the caller it was truncated.
        #
        # min_object_px must be *small* to overflow: it shrinks the crop, which
        # multiplies the tile count. An earlier version of this test passed 2000,
        # which is large enough to collapse the grid to the single full-frame tile,
        # so every assertion below passed without the overflow path ever running.
        plan = plan_tiles(4000, 3000, 640, 640, 8)
        self.assertEqual(len(plan), MAX_TILES, "expected the cap to bind here")
        self.assertTrue(tile_grid_overflowed(4000, 3000, 640, 640, 8))

    def test_crop_scale_never_puts_the_target_below_the_model_floor(self):
        """The invariant the whole module is built on, asserted rather than assumed.

        A min_object_px object inside a crop, scaled down to model input, must land
        at or above MIN_MODEL_PX. Rounding the crop size up instead of down breaks
        this by a fraction of a pixel, which is exactly the kind of drift no other
        test here would notice.
        """
        for img_w, img_h in [(1920, 1080), (2592, 1944), (3840, 2160)]:
            for min_object_px in range(10, 201, 7):
                plan = plan_tiles(img_w, img_h, 640, 640, min_object_px)
                for tile in plan:
                    if tile.full_frame:
                        continue  # tile 0 is the un-magnified baseline by design
                    scale = 640.0 / max(tile.width, tile.bottom - tile.top)
                    self.assertGreaterEqual(
                        min_object_px * scale, MIN_MODEL_PX,
                        f"{img_w}x{img_h} min_object_px={min_object_px}: a "
                        f"{min_object_px}px object reaches the model at "
                        f"{min_object_px * scale:.4f}px, under the {MIN_MODEL_PX}px floor",
                    )

    def test_overflow_predicate_false_for_normal_grids(self):
        self.assertFalse(tile_grid_overflowed(1920, 1080, 640, 640, 60))
        self.assertFalse(tile_grid_overflowed(640, 480, 640, 640, 60))

    def test_invalid_inputs_fall_back_to_full_frame(self):
        self.assertEqual(len(plan_tiles(1920, 1080, 640, 640, 0)), 1)
        self.assertEqual(len(plan_tiles(1920, 1080, 640, 640, -5)), 1)
        self.assertEqual(len(plan_tiles(0, 0, 640, 640, 60)), 1)

    def test_bad_overlap_raises(self):
        with self.assertRaises(ValueError):
            plan_tiles(1920, 1080, 640, 640, 60, overlap=1.0)
        with self.assertRaises(ValueError):
            plan_tiles(1920, 1080, 640, 640, 60, overlap=-0.1)


class TestSelectTiles(unittest.TestCase):
    def setUp(self):
        self.plan = plan_tiles(2592, 1944, 640, 640, 60)

    def test_budget_is_exact(self):
        for budget in (2, 3, 4):
            selected = select_tiles(self.plan, budget, 1.0, 1000.0)
            self.assertEqual(len(selected), min(budget, len(self.plan)))

    def test_content_never_changes_the_budget(self):
        # The invariant the design exists to protect: selection depends only on the
        # clock and the plan, never on anything scene-derived.
        sizes = {
            len(select_tiles(self.plan, 3, 1.0, float(t))) for t in range(0, 100)
        }
        self.assertEqual(sizes, {3})

    def test_full_frame_always_included(self):
        for t in range(0, 50):
            selected = select_tiles(self.plan, 3, 1.0, float(t))
            self.assertTrue(selected[0].full_frame)

    def test_selection_is_stateless_and_repeatable(self):
        a = select_tiles(self.plan, 3, 1.0, 1234.0)
        b = select_tiles(self.plan, 3, 1.0, 1234.0)
        self.assertEqual([t.index for t in a], [t.index for t in b])

    def test_no_duplicate_tiles_within_one_cycle(self):
        for t in range(0, 50):
            selected = select_tiles(self.plan, 3, 1.0, float(t))
            indices = [x.index for x in selected]
            self.assertEqual(len(indices), len(set(indices)))

    def test_every_tile_visited_within_the_sweep_bound(self):
        budget = 3
        cycles = sweep_cycles(len(self.plan), budget)
        seen = set()
        for step in range(cycles):
            for tile in select_tiles(self.plan, budget, 1.0, float(step)):
                seen.add(tile.index)
        self.assertEqual(
            seen, {t.index for t in self.plan},
            f"not all tiles visited in {cycles} cycles: missing "
            f"{ {t.index for t in self.plan} - seen }",
        )

    def test_no_starvation_over_a_long_run(self):
        counts = {t.index: 0 for t in self.plan}
        for step in range(200):
            for tile in select_tiles(self.plan, 3, 1.0, float(step)):
                counts[tile.index] += 1
        for index, count in counts.items():
            self.assertGreater(count, 0, f"tile {index} starved")

    def test_slow_caller_covers_everything_when_tile_period_matches(self):
        # A caller on lightNVR's keyframe-gated path fires at the GOP length rather
        # than the configured detection_interval. Told the truth via tile_period,
        # coverage is complete.
        seen = set()
        for step in range(0, 60, 2):  # fires every 2s, tile_period=2
            for tile in select_tiles(self.plan, 3, 2.0, float(step)):
                seen.add(tile.index)
        self.assertEqual(seen, {t.index for t in self.plan})

    def test_commensurate_period_mismatch_starves_tiles(self):
        # Pinned failure mode, not aspiration. The cursor advances linearly with the
        # clock, so a caller whose period is an exact multiple of tile_period samples
        # the same residues forever. Half this grid is never visited. tile_period is
        # the remedy — see the test above and select_tiles' docstring.
        seen = set()
        for step in range(0, 60, 2):  # fires every 2s but claims tile_period=1
            for tile in select_tiles(self.plan, 3, 1.0, float(step)):
                seen.add(tile.index)
        self.assertNotEqual(
            seen, {t.index for t in self.plan},
            "aliasing was expected here; if this now passes the cursor changed and "
            "the docstring's contract should be relaxed to match",
        )
        self.assertEqual(seen, {0, 1, 2})

    def test_budget_of_one_returns_only_the_full_frame(self):
        selected = select_tiles(self.plan, 1, 1.0, 5.0)
        self.assertEqual(len(selected), 1)
        self.assertTrue(selected[0].full_frame)

    def test_budget_larger_than_grid_is_clamped(self):
        selected = select_tiles(self.plan, 99, 1.0, 5.0)
        self.assertEqual(len(selected), len(self.plan))

    def test_collapsed_plan_is_handled(self):
        plan = plan_tiles(640, 480, 640, 640, 60)
        self.assertEqual(len(select_tiles(plan, 4, 1.0, 7.0)), 1)

    def test_empty_plan_is_handled(self):
        self.assertEqual(select_tiles([], 4, 1.0, 7.0), [])

    def test_sweep_bound_matches_the_plan_document(self):
        # K=24 pool with T=4 sweeps in 8 cycles: at 1 Hz that is 8 s, inside a
        # walking person's dwell time. This is the number the design rests on.
        self.assertEqual(sweep_cycles(25, 4), 8)


class TestCoordinateMapping(unittest.TestCase):
    def test_full_frame_tile_is_identity(self):
        tile = Tile(0, 0, 0, 1920, 1080, full_frame=True)
        box = (0.1, 0.2, 0.3, 0.4)
        mapped = map_tile_box_to_frame(box, tile, 1920, 1080)
        for got, want in zip(mapped, box):
            self.assertAlmostEqual(got, want, places=9)

    def test_round_trip_through_the_crop_rect_is_identity(self):
        tile = Tile(1, 320, 200, 1920, 1000)
        for box in [
            (0.0, 0.0, 1.0, 1.0),
            (0.25, 0.5, 0.75, 0.9),
            (0.4, 0.4, 0.45, 0.45),
        ]:
            frame_box = map_tile_box_to_frame(box, tile, 1920, 1080)
            # Invert: frame-normalized -> pixels -> tile-normalized.
            back = (
                (frame_box[0] * 1920 - tile.left) / tile.width,
                (frame_box[1] * 1080 - tile.top) / tile.height,
                (frame_box[2] * 1920 - tile.left) / tile.width,
                (frame_box[3] * 1080 - tile.top) / tile.height,
            )
            for got, want in zip(back, box):
                self.assertAlmostEqual(got, want, places=6)

    def test_offset_tile_shifts_the_box(self):
        tile = Tile(1, 960, 0, 1920, 1080)
        mapped = map_tile_box_to_frame((0.0, 0.0, 1.0, 1.0), tile, 1920, 1080)
        self.assertAlmostEqual(mapped[0], 0.5, places=9)
        self.assertAlmostEqual(mapped[2], 1.0, places=9)

    def test_results_are_clamped_to_the_unit_square(self):
        tile = Tile(1, 1600, 0, 1920, 1080)
        mapped = map_tile_box_to_frame((-0.5, -0.5, 1.5, 1.5), tile, 1920, 1080)
        for value in mapped:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_degenerate_frame_does_not_divide_by_zero(self):
        tile = Tile(0, 0, 0, 0, 0, full_frame=True)
        self.assertEqual(map_tile_box_to_frame((0, 0, 1, 1), tile, 0, 0), (0.0, 0.0, 0.0, 0.0))


class TestTruncation(unittest.TestCase):
    def test_box_touching_an_interior_tile_edge_is_truncated(self):
        tile = Tile(1, 0, 0, 1600, 1080)  # right edge is interior to a 1920 frame
        self.assertTrue(is_truncated((0.5, 0.2, 1.0, 0.8), tile, 1920, 1080))

    def test_box_touching_a_frame_edge_is_not_truncated(self):
        tile = Tile(1, 320, 0, 1920, 1080)  # right edge IS the frame edge
        self.assertFalse(is_truncated((0.5, 0.2, 1.0, 0.8), tile, 1920, 1080))

    def test_left_frame_edge_is_not_truncated(self):
        tile = Tile(1, 0, 0, 1600, 1080)
        self.assertFalse(is_truncated((0.0, 0.2, 0.4, 0.8), tile, 1920, 1080))

    def test_interior_box_is_not_truncated(self):
        tile = Tile(1, 0, 0, 1600, 1080)
        self.assertFalse(is_truncated((0.3, 0.3, 0.6, 0.6), tile, 1920, 1080))

    def test_vertical_interior_edge_is_truncated(self):
        tile = Tile(1, 0, 0, 1600, 800)  # bottom edge interior to a 1080 frame
        self.assertTrue(is_truncated((0.3, 0.6, 0.6, 1.0), tile, 1920, 1080))


class TestMerge(unittest.TestCase):
    def test_duplicate_across_overlap_collapses_to_one(self):
        a = TileDetection("person", 0.9, (0.50, 0.50, 0.60, 0.70), tile_index=1)
        b = TileDetection("person", 0.8, (0.505, 0.505, 0.605, 0.705), tile_index=2)
        merged = merge_tile_detections([a, b])
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].confidence, 0.9)

    def test_distinct_objects_both_survive(self):
        a = TileDetection("person", 0.9, (0.1, 0.1, 0.2, 0.3), tile_index=1)
        b = TileDetection("person", 0.8, (0.7, 0.6, 0.8, 0.9), tile_index=2)
        self.assertEqual(len(merge_tile_detections([a, b])), 2)

    def test_different_labels_do_not_suppress_each_other(self):
        a = TileDetection("person", 0.9, (0.5, 0.5, 0.6, 0.7), tile_index=1)
        b = TileDetection("car", 0.8, (0.5, 0.5, 0.6, 0.7), tile_index=2)
        self.assertEqual(len(merge_tile_detections([a, b])), 2)

    def test_truncated_copy_loses_to_the_interior_copy(self):
        # Same object: clipped at a tile seam in one tile, whole in another. The
        # whole copy must win even though the fragment scored higher.
        fragment = TileDetection("person", 0.95, (0.50, 0.50, 0.55, 0.70),
                                 tile_index=1, truncated=True)
        whole = TileDetection("person", 0.60, (0.50, 0.50, 0.60, 0.70),
                              tile_index=2, truncated=False)
        merged = merge_tile_detections([fragment, whole])
        self.assertEqual(len(merged), 1)
        self.assertFalse(merged[0].truncated)
        self.assertEqual(merged[0].tile_index, 2)

    def test_small_fragment_is_suppressed_by_containment_despite_low_iou(self):
        # A sliver of a large object has poor IoU against the full box, so plain NMS
        # would keep it as a phantom. Containment is what catches this.
        whole = TileDetection("car", 0.9, (0.20, 0.20, 0.80, 0.80), tile_index=1)
        sliver = TileDetection("car", 0.85, (0.20, 0.20, 0.28, 0.80),
                               tile_index=2, truncated=True)
        self.assertLess(box_iou(whole.box, sliver.box), 0.45)
        merged = merge_tile_detections([whole, sliver])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].tile_index, 1)

    def test_untruncated_sliver_is_kept(self):
        # Containment only overrides for truncated boxes: a genuinely small object
        # sitting inside a larger one's box is real and must survive.
        big = TileDetection("car", 0.9, (0.2, 0.2, 0.8, 0.8), tile_index=1)
        small = TileDetection("car", 0.7, (0.3, 0.3, 0.36, 0.4),
                              tile_index=1, truncated=False)
        self.assertEqual(len(merge_tile_detections([big, small])), 2)

    def test_output_is_confidence_ordered(self):
        dets = [
            TileDetection("person", 0.30, (0.0, 0.0, 0.05, 0.05), tile_index=1),
            TileDetection("person", 0.90, (0.2, 0.2, 0.25, 0.25), tile_index=1),
            TileDetection("person", 0.60, (0.4, 0.4, 0.45, 0.45), tile_index=2),
        ]
        merged = merge_tile_detections(dets)
        confidences = [d.confidence for d in merged]
        self.assertEqual(confidences, sorted(confidences, reverse=True))

    def test_empty_input(self):
        self.assertEqual(merge_tile_detections([]), [])

    def test_single_detection_passes_through(self):
        d = TileDetection("person", 0.5, (0.1, 0.1, 0.2, 0.2), tile_index=0)
        self.assertEqual(len(merge_tile_detections([d])), 1)


class TestBoxMath(unittest.TestCase):
    def test_iou_of_identical_boxes_is_one(self):
        b = (0.1, 0.1, 0.5, 0.5)
        self.assertAlmostEqual(box_iou(b, b), 1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        self.assertEqual(box_iou((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.6, 0.6)), 0.0)

    def test_iou_half_overlap(self):
        a = (0.0, 0.0, 0.2, 0.1)
        b = (0.1, 0.0, 0.3, 0.1)
        # intersection 0.1x0.1, union 0.3x0.1 -> 1/3
        self.assertAlmostEqual(box_iou(a, b), 1.0 / 3.0, places=9)

    def test_containment_of_fully_inside_box_is_one(self):
        inner = (0.3, 0.3, 0.4, 0.4)
        outer = (0.1, 0.1, 0.9, 0.9)
        self.assertAlmostEqual(containment(inner, outer), 1.0)

    def test_containment_of_disjoint_box_is_zero(self):
        self.assertEqual(containment((0.0, 0.0, 0.1, 0.1), (0.5, 0.5, 0.9, 0.9)), 0.0)

    def test_containment_of_zero_area_box_is_zero(self):
        self.assertEqual(containment((0.5, 0.5, 0.5, 0.5), (0.1, 0.1, 0.9, 0.9)), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
