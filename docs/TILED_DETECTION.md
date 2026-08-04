# Tiled detection for small objects

Distant objects are often too small to detect once a frame has been scaled down to
the model's input. A 1080p frame reaching a 640 px YOLO model is scaled by 0.333,
so a 60 px person arrives at 20 px — below the ~24 px where the model reliably
fires.

Tiled detection crops regions at a higher effective scale and rotates through them
on a fixed budget. **Per request it runs exactly `tiles` inferences.** Scene content
never changes that number — it only changes which regions get inspected — so cost is
flat whether the frame is empty or a tree is thrashing in the wind.

Tile 0 is always the whole frame, so large and near objects are still caught every
cycle and behaviour never regresses. The remaining `tiles - 1` rotate over the grid
on a clock-derived cursor, which needs no server-side state.

## Parameters

Every option can be set two ways: as a **query parameter** on `/api/v1/detect` for
per-request control, or as an **environment variable** to change the default for the
whole server. The query parameter wins when both are present.

| Query parameter | `.env` variable | Default | Meaning |
|---|---|---|---|
| `tiles` | `TILE_BUDGET` | `1` | Inferences per request. `1` disables tiling entirely. |
| `min_object_px` | `TILE_MIN_OBJECT_PX` | `60` | Smallest object to resolve, in **source** pixels. Drives crop size. |
| `tile_overlap` | `TILE_OVERLAP` | `0.25` | Overlap between adjacent crops, as a fraction of crop size. |
| `tile_period` | `TILE_PERIOD_SECONDS` | `1.0` | Rotation period in seconds. **Must match the rate the caller actually fires at** — see below. |
| `tile_deadline_s` | `TILE_DEADLINE_SECONDS` | `7.0` | Stop issuing tiles once this many seconds have elapsed. |
| `stream` | — | — | Stream name, logged for correlation. Per-request only. |
| — | `TILE_IOU_THRESHOLD` | `0.45` | IoU threshold for cross-tile NMS. Server-wide only; no query parameter. |

Note the name changes between the two forms: `min_object_px` is `TILE_MIN_OBJECT_PX`,
`tile_period` is `TILE_PERIOD_SECONDS`, and `tile_deadline_s` is
`TILE_DEADLINE_SECONDS`. The query names are shorter because they are typed into
lightNVR stream URLs by hand.

The 32-tile ceiling is the module constant `MAX_TILES` in `utils/tiling.py` and is not
configurable from either place.

## Setting defaults in `.env`

Settings load from a `.env` file in the working directory (`config.py`, via
`pydantic-settings`). Names are **case-sensitive** and must be upper-case exactly as
listed above:

```dotenv
# Tiled detection
TILE_BUDGET=4
TILE_MIN_OBJECT_PX=60
TILE_OVERLAP=0.25
TILE_PERIOD_SECONDS=1.0
TILE_DEADLINE_SECONDS=7.0
TILE_IOU_THRESHOLD=0.45
```

In Docker the file is mounted at `/app/.env`:

```yaml
volumes:
  - ./lod.env:/app/.env:ro
```

These values are read once at import time and become the query parameters' defaults,
so **changing `.env` requires a container or process restart** to take effect. Query
parameters need no restart, which is why per-camera tuning belongs in the stream URL.

## Usage

```bash
curl -X POST "http://localhost:8000/api/v1/detect?tiles=4&min_object_px=60&tile_period=1" \
  -F "file=@frame.jpg"
```

## Choosing `min_object_px`

This is the only setting that really matters. It is the smallest thing you want
detected, measured in pixels *in the source frame* — so photograph the scene and
measure a person at the distance you care about. Lower values magnify more but
produce more tiles, which lengthens the rotation:

| Frame | `min_object_px=60` | `=40` | `=24` (native) |
|---|---|---|---|
| 1080p | 3 tiles | 4 | 9 |
| 5 MP | 5 tiles | 10 | 25 |
| 4K | 7 tiles | 16 | 32 (capped) |

If the grid exceeds 32 tiles it is truncated and a warning is logged — treat that as
a signal that `min_object_px` is set finer than the frame can sweep in reasonable time.

## Choosing `tiles`

It is a per-request parameter, so set it per camera rather than globally. Close-range
cameras can run `tiles=1` and still benefit from full-resolution detection. Spend the
budget on the distant views. As a rough guide, on an Intel HD 530 with the OpenVINO
provider a 640 px YOLO inference costs ~30 ms, so six cameras at 1 Hz and `tiles=4` is
around 70% of that GPU. Passing a `tiles` larger than the grid is harmless — the extra
budget is simply not used.

## `tile_period` must match your real detection rate

The rotation cursor is derived from the clock, so a caller whose period is an exact
multiple of `tile_period` will sample the same tiles forever and never visit the
others — permanent blind spots, not merely a slower sweep. This matters because
lightNVR's keyframe-gated path fires at the GOP length, not at the configured
`detection_interval`. If a stream is gated on keyframes with a 2 s GOP, set
`tile_period=2`.

## Deadline behaviour

If the deadline is reached, remaining tiles are skipped and partial results returned.
Tile 0 always runs first, so a slow or degraded host degrades to ordinary whole-frame
behaviour rather than returning nothing.

## Per-stream configuration from lightNVR

lightNVR passes the API URL through verbatim, including any query string, so every
option above can be set per stream from the stream's custom-endpoint field — no
lightNVR changes required:

```
http://<docker-host>:8000/api/v1/detect?stream=driveway&filter_classes=person,car&tiles=4&min_object_px=60&tile_period=1
```

Unrecognised parameters are ignored, so a URL written for a newer server stays safe
against an older one.
