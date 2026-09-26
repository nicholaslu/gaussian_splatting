# Tile rasterisation experiment

This branch tries rasterising the prepared splats in compute, as the 3DGS
reference rasteriser and gsplat do: one workgroup per 16x16 tile of pixels,
the splats composited front to back, each pixel stopping once it is opaque.
It is not merged, because on both backends it is slower than the render
pipeline the display uses.

## What it does

GPU view preparation runs unchanged and leaves the draw stream in depth order.
Six stages then follow, in both `gsplat_prepare.metal` and
`GLSL430/gsplat_prepare.comp`:

1. `raster_bin_count` and `raster_bin_scan` count the tiles each quad touches
   and where each splat's pairs start.
2. `raster_bin_emit` writes one (tile, draw position) pair per tile, in draw
   order.
3. The existing radix sort sorts the pairs on the tile id alone. Because it is
   stable, each tile's pairs stay in depth order.
4. `raster_clear_ranges` and `raster_tile_ranges` find each tile's run of
   pairs.
5. `raster_tiles` composites each tile. Each 32-wide SIMD group holds an 8x4
   block of pixels, and first lists which splats of a batch reach its block.

`test/bench_tile_raster.mm` (Metal) and `test/bench_tile_raster_gl.cpp` (EGL)
load a trained scene with its `cameras.json` through `test/real_scene.hpp`. They
draw the same prepared instances three ways: through the display's quad
shaders into RGBA8, into RGBA16F, and with the tile rasteriser. They then
compare the images and the GPU times.

## Results

Full Garden scene (5,834,784 splats) at 2940x1905, median GPU time in ms.
"Close" moves training camera 0 3.5 units forward, onto the table.

| | quads, RGBA8 | quads, RGBA16F | tile rasteriser |
|---|---:|---:|---:|
| Apple M3, camera 0 / 40 | 33.5 / 31.8 | 34.9 / 31.0 | 56.9 / 55.9 |
| Apple M3, close | 51.4 | 49.2 | 97.4 |
| RTX 4090, camera 0 / 40 | 2.64 / 2.39 | 2.98 / 2.97 | 7.86 / 6.28 |
| RTX 4090, close | 4.14 | 6.82 | 11.0 |

Why the tile rasteriser loses:

- **Most splats are small.** The splats are a few pixels across, so every pixel
  tested 826 splats in its tile's range while only 62 covered it. Listing per
  8x4 block recovered 10-15%.
- **Stopping early saves little.** About half the pixels end opaque, and 51
  splats contribute per pixel against the 62 that cover it.
- **Binning and sorting alone cost too much.** On the 4090 they already cost
  1.5-4 ms, about what the render pipeline takes for the whole draw. On the M3
  the tile sort alone is 10-28 ms.

## Accuracy

Against quads blended into RGBA16F, per image:

| | PSNR | pixels more than 2 steps off |
|---|---:|---:|
| quads blended into RGBA8 | 40-50 dB | 1-17% (M3), 10-40% (4090) |
| tile rasteriser | 57-63 dB | about 0.01% |

The tile rasteriser accumulates in float, as the reference does. The RGBA8
target rounds after every one of the dozens of layers a pixel blends.

Blending into a half float target recovers this accuracy without the tile
rasteriser. On the M3 it costs nothing measurable, and on the 4090 it adds
0.3-2.7 ms.
