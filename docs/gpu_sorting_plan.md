# GPU view-preparation plan

## Problem

Every meaningful camera movement currently runs `sortIndexBuffer()` on the
render thread. That function does substantially more than sorting:

1. It visits every splat and rejects transparent, out-of-frustum, and too-small
   splats.
2. It evaluates spherical-harmonic colour for every survivor.
3. It creates a 64-bit depth-key/index pair and concatenates the worker output.
4. It performs four CPU radix passes.
5. It gathers a sorted 16-byte `DrawInstance` stream and uploads it.

The radix conversion was already valuable: at 616,187 visible splats on an M3,
the historical comparison sort took 229.3 ms, while the radix sort took 8.3 ms.
The complete CPU preparation still took 51.0 ms at that point. Later commits
parallelised culling, SH evaluation, and gathering and reduced the instance
record, but a multi-million-splat Garden view still produces a tens-of-ms
camera-movement spike.

The first full Garden preparation after those changes, on the current
development machine with 5,834,784 total and 1,647,869 visible splats, measured
80.248 ms:

| Stage | Time | Share |
| --- | ---: | ---: |
| Setup | 0.021 ms | 0.0% |
| Cull and SH | 60.933 ms | 75.9% |
| Merge worker partitions | 1.573 ms | 2.0% |
| Radix sort | 11.600 ms | 14.5% |
| Gather instances | 4.187 ms | 5.2% |
| Upload instance buffer | 1.932 ms | 2.4% |

This confirms that “CPU preparation” must not be interpreted as pure sorting.
Even a zero-cost sort would leave roughly 68.6 ms in the other stages for this
sample. The first sample includes initial worker-partition capacity allocation,
so a camera-movement sample with reused buffers may be somewhat lower. The GPU
work should nevertheless move culling and SH/gather as well as radix.

Moving only step 4 to the GPU is not sufficient. Reading the sorted indices
back would introduce a GPU-to-CPU dependency, while steps 1, 2, 3, 5, and the
upload would remain. The useful boundary is the whole per-view preparation.

## Target pipeline

Static scene data and SH coefficients are uploaded once. On a camera update:

1. A compute pass performs culling, projected-radius testing, and depth-key
   generation.
2. Survivors are compacted into key/index buffers. An atomic counter records
   the visible count. SH is deliberately not evaluated yet, so colour data is
   not carried through every radix pass.
3. A GPU radix sort orders the key/index pairs, preserving the current CPU
   back-to-front order exactly.
4. A gather pass follows the sorted indices, evaluates SH only for visible
   splats, and writes the final `DrawInstance` buffer.
5. The GPU writes the instance count into indirect draw arguments and renders
   without a CPU readback.

The first renderer should retain the current global-order quad rasterisation.
That isolates scheduling and ordering changes from a more fundamental rendering
algorithm change. Tile binning and tile-local sorting can be evaluated later.

## Backend findings

### Metal

Metal has all primitives required for this pipeline: compute command encoders,
writable buffers, ordered passes in a command buffer, and indirect draw
arguments. The blocker is the current Ogre 1.12 Metal backend, not Metal.

The public Ogre API exposes compute programs and `_dispatchCompute()`, and the
Metal backend can compile a compute function and create a compute encoder.
However, this fork currently:

- advertises `RSC_COMPUTE_PROGRAM` but does not override `_dispatchCompute()`;
- treats a compute program as a fragment program in `bindGpuProgram()`;
- sends compute constants through `setVertexBytes()` on the render encoder;
- does not advertise UAV or atomic-counter support.

Consequently, adding a compute program to the plugin alone cannot work. The
preferred implementation is to complete the Ogre Metal compute path first:
compute pipeline binding, compute constant/resource binding, dispatch, buffer
barriers/lifetime, and writable buffer exposure. This keeps command-buffer
ownership and render ordering inside Ogre. A native Objective-C++ Metal path in
the plugin is a viable prototype, but it couples the plugin to Ogre Metal
internals and is not the long-term boundary.

There are two Metal sorting candidates to benchmark before committing to one:

- `MPSGraph.argSort` is an Apple-provided GPU operation that returns 32-bit
  indices. It can quickly establish whether GPU sorting removes the observed
  spike, but its behaviour and temporary-memory cost for a changing compacted
  length must be measured. Its graph encoder may also commit and continue the
  supplied command buffer, which conflicts with assuming Ogre retains sole
  ownership of that buffer.
- A custom stable LSD radix sort gives exact control over compacted input,
  memory reuse, pass timing, and command-buffer ordering. A straightforward
  implementation uses histogram, prefix-scan, and stable-scatter stages for
  each digit. Start with the same four 8-bit digits as the CPU reference; use
  fewer key bits only after image comparisons show that depth quantisation is
  acceptable.

#### Sorting microbenchmarks

Two standalone benchmarks now live in `tools/`. They wait for GPU completion
for timing, keep their result buffers GPU-resident during the timed interval,
and validate the complete key/index result afterward. Measurements below are
on the current Apple M3 development machine; values are steady-run medians.

| Elements | MPSGraph float32 arg-sort | Custom 32-bit radix | Custom 16-bit radix |
| ---: | ---: | ---: | ---: |
| 500,000 | 1.60 ms | 2.15 ms | 0.82 ms |
| 1,000,000 | 3.14 ms | 4.18 ms | 1.69 ms |
| 1,650,000 | 4.77 ms | 5.77 ms | 1.93 ms |
| 3,000,000 | 12.31 ms | 10.96 ms | 3.58 ms |
| 5,800,000 | 27.20 ms | 23.20 ms | 9.39 ms |

MPSGraph is a useful performance oracle, but it is not the preferred renderer
path. Its encoded graph may commit and continue the supplied command buffer,
and a graph has a fixed input shape; compacting a variable survivor count before
sorting it would otherwise require a CPU readback or sorting the full capacity.
The custom radix consumes an explicit count buffer and can stay on Ogre's Metal
command timeline. Its tile/histogram/scatter structure is adapted from the MIT-
licensed [MetalSprocketsGaussianSplats GPU sorter](https://github.com/schwa/MetalSprocketsGaussianSplats),
with the license retained in `gaussian_splatting_rviz_plugins/THIRD_PARTY_NOTICES.md`.

The exact 32-bit radix reduces the measured 1.65-million-key CPU radix from
11.60 ms to about 5.77 ms. A 16-bit half-depth key needs only two passes and
reduces that to about 1.93 ms, but it changes ordering precision. It is therefore
an optional candidate, not the default, until deterministic image comparisons
show that equal/quantised depth ties do not produce visible blending artifacts.

These are sort-only numbers. They do not alter the earlier conclusion that the
whole GPU path must include culling, compaction, SH evaluation, gather, and an
indirect draw: the measured CPU cull/SH stage is still the dominant 60.93 ms.

### OpenGL

Both Ogre's GL3Plus backend and a sufficiently modern compatibility context
provide compute shaders and shader-storage buffers. RViz currently loads the
legacy `RenderSystem_GL`, so this implementation uses its compatibility context
directly: GLSL 1.20 remains available for RViz's raster materials while GLSL
4.30 performs preparation. macOS stops at OpenGL 4.1 and retains the CPU
implementation. GPU sorting is therefore a capability-selected acceleration,
not a replacement for the portable path.

The implementation now targets the modern compatibility context exposed by
Ogre's `RenderSystem_GL` on Linux. This preserves RViz's existing
GLSL 1.20 raster materials while running preparation from GLSL 4.30 compute:
frustum and screen-size culling, stable compaction, four-pass 32-bit radix,
SH evaluation, projection, gather, and a GPU-written indirect instance count.
It uses no subgroup extension: stable ranks inside each 32-lane tile are
formed in shared memory, so core OpenGL 4.3 is the real minimum. A three-slot
fenced readback reports the visible count asynchronously and is not on the
draw dependency chain. macOS does not compile this backend because its OpenGL
implementation stops at 4.1.

Windows remains on the CPU fallback for now. Ogre compiles GLEW privately into
`RenderSystem_GL.dll`, so the plugin needs a small exported native-function
loader bridge before the same compute adapter can link there safely.

Backend selection is centralised behind `GpuViewPreparation`. In addition to
automatic capability selection, `GSPLAT_GPU_BACKEND` accepts `metal`,
`opengl`, `cuda`, and `cpu`. `cuda` currently reports that it is not built;
the selection point and factory contract are reserved so a CUDA/CUB OpenGL
interop backend can be added without changing the display or its projected
instance format.

### CUDA

On NVIDIA systems, CUB `DeviceRadixSort::SortPairs` was the candidate optional
backend, sorting OpenGL's buffers in place through CUDA/OpenGL interop. A
headless prototype on an RTX 4090 (CUDA 12.4) measured it for the Garden scale:
CUB sorts 5,834,784 key/value pairs in 0.34 ms, and with the interop map,
splitting the interleaved pairs, sorting and merging back it takes 0.59 ms a
frame, or 0.47 ms with separate key and value buffers. CUB's element count is a
host value, so the sort would run over the full capacity with the culled tail
filled with maximal keys (0.02 ms); its order matched the OpenGL radix entry for
entry, ties included. Against the OpenGL sort of the time, 7.0 ms, that was a
large saving, but after the parallel scans and subgroup ballots the OpenGL sort
takes 1.40 ms on the same GPU, and CUDA would save less than a millisecond a
frame for a build dependency and a second API on the draw path. It is not
pursued; the prototype and its report are on Shannon's `cuda-sort` branch.

The reference 3DGS rasterizer goes further: it emits tile/depth keys, sorts
key/value pairs with CUB, identifies each tile's sorted range, and blends tiles
on the GPU. The `tile-raster-experiment` branch tries that rasteriser in
compute on Metal and OpenGL, after this preparation; see
`docs/tile_rasterisation_experiment.md` there. It is 1.7-3x slower than drawing
the quads through the render pipeline on both an Apple M3 and an RTX 4090, close
views included, because most Garden splats cover a few pixels of a 16x16 tile
and only about half the pixels become opaque, so stopping early saves little.

## Implementation stages

### 0. Measurement and correctness harness

- Split the existing status timing into cull/SH, concatenate, radix, gather,
  and upload values.
- Record total and visible counts for Bonsai and Garden at fixed camera poses.
- Add a CPU reference test for depth-key order, culling, and SH output.
- Capture image comparisons for the CPU and future GPU paths.
- Benchmark `MPSGraph.argSort` and a small custom radix prototype on 0.5, 1,
  3, and 5.8 million keys without any CPU readback. Record GPU duration and
  peak temporary allocation; do not infer either from the RViz frame rate.

### 1. Interaction fallback

Add an optional sorting budget while the full GPU path is built: reuse the last
order during small camera deltas and cap sorting to a configurable frequency,
then always perform a final exact sort after motion stops. This removes the
worst UI stalls but can temporarily produce blending-order errors, so it should
not become the default correctness path without visual evaluation.

### 2. Complete Ogre Metal compute support

- Add a real active compute program and compute pipeline state.
- Bind constants, textures, read-only buffers, writable buffers, and counters
  to the compute encoder.
- Implement `_dispatchCompute()` and the render/compute encoder transitions.
- Make a compute-written Ogre vertex/instance buffer consumable by the later
  render pass without CPU synchronization.
- Add a minimal integration test whose compute pass writes vertices that are
  drawn in the same frame.

### 3. GPU key generation and global radix sort

- Upload positions and SH coefficients into GPU-readable buffers.
- Implement cull/key generation and compaction.
- Implement a stable four-pass 8-bit radix sort with ping-pong buffers, matching
  `depthKey()` and the CPU ordering.
- Evaluate SH and gather the sorted draw stream on the GPU, then use a
  GPU-generated visible count. Do not read results back during rendering.
- Keep the current CPU path as a runtime fallback and a validation reference.

### 4. Optional backends and algorithmic renderer

- Port the same buffer contract to GL 4.3 compute.
- Add a CUB backend for CUDA-capable NVIDIA/OpenGL systems if the deployment
  benefit justifies the dependency and interop complexity.
- Prototype tile binning and tile-local sorting only after the global GPU path
  is correct. It can reduce ordering work and overdraw, but it replaces the
  current Ogre draw with a more specialised rasterizer.

## Resource budget

For 5.8 million splats, one 64-bit key/index buffer is about 44.5 MiB. A radix
sort needs two such buffers. One final 16-byte draw buffer is about 89 MiB at
the worst case where every splat is visible. Static degree-3 SH data is much
larger (about 1 GiB at this count), so the first GPU implementation must
explicitly budget whether SH data is uploaded as float32, stored at half
precision, or shared with the ROS message without another full-size copy. SH
should be read only during the post-sort gather, for visible splats. This memory
decision must be measured for both macOS and iPadOS.

As built, preparation keeps the two key/index buffers, the per-tile tables,
two buffers of 64-byte `DrawnSplat`s - the survivors by compacted slot and in
draw order - and, for Colour Refresh, a colour and a slot's splat and draw
position per splat, all sized for every splat: about 950 MiB for Garden, beside the 48-byte
projected instance buffer the draw reads (267 MiB) and the scene itself
(records 267 MiB, SH 1.1 GiB). Before draw orders were kept, preparation took
about 400 MiB. The offscreen target is 8 bytes a pixel in half float.

## Acceptance criteria

- Camera-motion CPU preparation is below 2 ms on the render thread for full
  Garden; there is no per-frame GPU readback.
- GPU timing is reported separately for cull/SH, sort, and gather.
- CPU and GPU paths agree on visible count and depth order for deterministic
  test cameras, allowing only documented floating-point colour differences.
- A stationary camera does no repeated preparation.
- Metal uses the GPU path; legacy OpenGL remains correct through the CPU path;
  GL 4.3 and CUDA are capability-selected additions rather than build-time
  requirements.

## Implementation status

The Metal path now follows the target pipeline end to end, with one change from
the first prototype: survivors are compacted before the sort.

- `make_depth_keys` marks a rejected splat in the index half of its record, so
  keys stay exact. `compact_count`, `compact_scan` and `compact_scatter` pack the
  survivors, in index order, into the front of the key buffer, and write the
  visible count, the indirect dispatch sizes and the draw arguments.
- The radix and the gather size themselves to the survivors through indirect
  dispatch. Nothing is read back; the visible count reaches the status panel
  from a command buffer completion handler, one frame late.
- `MetalViewPreparationCore` holds the pipelines, buffers and dispatch sequence
  with no Ogre dependency. The display's Ogre adapter and
  `test/verify_gpu_preparation.mm` both encode through it.
- The CPU path and the kernels read one `ViewParameters` block and share
  `prepareSplat()` in `splat_view.hpp`, which follows Ogre's evaluation order so
  the CPU path is bit-identical to the Ogre-based loop it replaced.
- The gather also projects each survivor, once per splat, with the matrices,
  viewport and field of view the vertex program used to take as auto
  parameters (`ProjectionParameters`). It writes the NDC centre, the scaled
  ellipse axes and the compensated colour, so `gsplat_projected_vp` only places
  the corners. Because the projection is baked in, GPU mode prepares again on
  any exact change of view rather than past the sort thresholds; `projectSplat()`
  in `splat_view.hpp` is its CPU reference. The CPU path keeps its instance
  format and the per-vertex projection in `gsplat_vp`.
- The eigen decomposition in every copy (`gsplat_vp`, `gsplat.vert`, the gather)
  now avoids two cancellations inherited from the reference: the eigenvalue
  spread is `sqrt(((a - c) / 2)^2 + b^2)` rather than `sqrt(mid^2 - det)`, and
  the major axis comes from the row of `Sigma - lambda1 I` that does not subtract
  nearly equal numbers. Nearly axis-aligned ellipses had been swinging by
  degrees under float rounding.

Measured on the M3 on synthetic scenes (uniformly random splats, not the real
Garden data), 5,834,784 splats at degree 3 with 2,285,046 visible, median of 7:

| Stage | Prototype | Compacted | Projection in gather |
| --- | ---: | ---: | ---: |
| Keys (and compaction) | 3.59 ms | 5.39 ms | 5.34 ms |
| Radix sort | 16.12 ms | 8.05 ms | 7.98 ms |
| Gather and SH (and projection) | 22.11 ms | 22.06 ms | 26.38 ms |
| One command buffer, as shipped | 41.74 ms | 35.42 ms | 39.78 ms |
| CPU encode | 0.019 ms | 0.018 ms | 0.017 ms |

Compaction halves the sort. The gather did not move, so its cost is SH
evaluation for the visible splats rather than dispatching threads that have
nothing to do; it is now the largest stage, and the total still exceeds a 60 Hz
frame at this scale.

Moving the projection into the gather adds about 4 ms there and removes the
per-vertex covariance, eigen decomposition and texture reads from the vertex
program, which had been the larger cost while moving (about 2.8 ns per vertex,
four vertices per splat). In RViz on the Garden scene, moving the camera now
starts at 60 fps where it had been 30 to 50, and settles at 20 to 25 fps once
the fanless M3 throttles under sustained load.

Against the acceptance criteria:

- CPU preparation below 2 ms with no readback: met; encoding takes 0.018 ms.
- GPU timing per stage: available with `GSPLAT_PROFILE_GPU` set, which runs the
  stages in command buffers of their own and waits for each. That stalls the
  render thread and is for measurement only.
- CPU/GPU agreement: `test/verify_gpu_preparation.mm`, registered with CTest,
  requires identical visible counts, draw arguments and depth order including
  ties, colours within 1e-4 (observed at most 7.2e-7), and every projected quad
  within float rounding: centres and ellipses compared in pixels, opacity and
  radius within 1e-3 relative (the compensation's own cancellation), worst case
  0.25 of tolerance. It covers tile-edge
  counts, heavy ties, orthographic and infinite-far views, culling off, an empty
  view, buffer reuse under a narrower view, in-place re-upload and a splat at
  the camera. `--quick` skips the Garden-sized scene.
- A stationary camera does no preparation: unchanged.
- Backend selection: Metal and OpenGL 4.3 use the common GPU-preparation
  contract, and any capability, allocation, compile, or dispatch failure falls
  back to the CPU path. OpenGL performs culling, stable compaction, four radix
  passes, SH, projection, gather, indirect dispatch, and indirect draw without
  a synchronous readback. CUDA has a reserved factory and buffer contract but
  no implementation yet.

Colour is evaluated before the sort. `shade_visible` (Metal) and
`GSPLAT_STAGE_SHADE_VISIBLE` (OpenGL) run between compaction and the radix, while
the survivors are still in index order, so the SH coefficients are read in the
order they are stored and the result is written per splat for the gather. The
gather used to evaluate SH itself after the sort, which read the coefficient
buffer in depth order, 180 bytes per splat scattered over about 1 GiB at Garden
scale. On Linux with an RTX 4090 and the full Garden PLY at 1420x719, timed
under `GSPLAT_PROFILE_GPU` as the median of 25 preparations after 5 warm-up:

| View | Visible | SH in gather | SH before sort | Cull+compact | Shade | Sort | Gather |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Main | 1,654,413 | 667.8-682.6 ms | 28.71 ms | 1.10 | 1.18 | 2.34 | 24.05 |
| Second | 1,470,244 | 590.3 ms | 25.90 ms | 1.10 | 1.08 | 2.18 | 21.20 |

The SH arithmetic takes 1.1-1.2 ms of it, so nearly all of the old cost was the
scattered reads. Both views render pixel-identical to the baseline. These are
GPU preparation times, measured synchronously, not frame intervals. On the M3,
synthetic Garden-sized scene, median of 7:

| Stage | SH in gather | SH before sort |
| --- | ---: | ---: |
| Cull and compaction | 5.43 ms | 5.47 ms |
| Shade | - | 9.58 ms |
| Radix sort | 8.08 ms | 8.22 ms |
| Gather (and SH) | 26.40 ms | 9.05 ms |
| One command buffer, as shipped | 40.10 ms | 32.02 ms |

Projection now happens before the sort as well. `shade_project` (Metal) and
`GSPLAT_STAGE_SHADE_PROJECT` (OpenGL) colour and project each survivor into its
compacted slot and make the slot the sort's value, so the gather only permutes
a buffer the size of the visible set rather than reading records and colours
scattered over the whole scene. The sort is stable and slots follow index
order, so the draw order is unchanged. The projected scratch costs 48 bytes per
splat, 267 MiB at Garden scale, in place of the 89 MiB colour buffer; it is the
size of the instance buffer, so it adds no new storage-block limit. On Metal,
`slotSplatBuffer()` maps slots back to splat indices for diagnostics. Same
synthetic scene, median of 7:

| Stage | M3, SH before sort | M3, projection before sort | RTX 4090, projection before sort |
| --- | ---: | ---: | ---: |
| Cull and compaction | 5.47 ms | 5.35 ms | 1.70 ms |
| Shade (and project) | 9.58 ms | 10.43 ms | 1.80 ms |
| Radix sort | 8.22 ms | 8.02 ms | 7.06 ms |
| Gather | 9.05 ms | 4.06 ms | 1.93 ms |
| Whole preparation | 32.02 ms | 27.83 ms | 13.00 ms |

OpenGL now has the structure Metal has. `OpenGlViewPreparationCore` holds the
programs, buffers and dispatch sequence with no Ogre in it; the display's
adapter supplies Ogre's instance and indirect buffers, and
`test/verify_gl_preparation.cpp` drives the core headless through EGL on the
cases `verify_gpu_preparation.mm` runs, sharing its scenes and CPU reference
through `test/preparation_test_scenes.hpp`. Besides exact order, keys, colours
and quads it checks that the GPU sorted its own keys stably and, for stock Ogre,
that instance slots past the visible count are empty. Its first run found the
depth keys a quarter of the time an ulp or more from the CPU's, by up to 46,000
ulp where the depth cancels to near zero: GLSL had contracted dot() into FMAs,
which a baseline x86-64 build of the CPU path does not. The culling and key
arithmetic now follows prepareSplat() term by term under `precise`, and every
case agrees exactly on the RTX 4090. A CPU build that does contract, as GCC
does for AArch64, can still differ in the last ulp. OpenGL stages are timed
with timestamp queries under `GSPLAT_PROFILE_GPU`, like Metal, and bind at most
eight storage blocks, the most OpenGL guarantees.

The scans are parallel. Timestamps inside the sort on the RTX 4090 put 4.2-5.4
ms of its 5-7 ms in the prefix sums, and 0.55 ms more in compaction's: 256
threads each walking a whole row of per-tile counts, and single threads summing
the digit totals and the tile counts. All three are now 256-thread workgroup
scans (Hillis-Steele, in core OpenGL 4.3 and plain Metal), walking a row 256
entries at a time. Same synthetic scene, median of 7, exact against the CPU on
both:

| Stage | RTX 4090 before | RTX 4090 after | M3 before | M3 after |
| --- | ---: | ---: | ---: | ---: |
| Cull and compaction | 1.70 ms | 1.21 ms | 5.35 ms | 4.97 ms |
| Radix sort | 5.09-6.71 ms | 1.53 ms | 8.02 ms | 5.97 ms |
| Whole preparation | 12.6 ms | 6.9-7.2 ms | 27.83 ms | 25.2-25.8 ms |

On the RTX 4090 the sort now splits into histograms 0.28 ms, scans 0.17 ms and
scatters 1.08 ms over the four passes; `GSPLAT_PROFILE_GPU` and the OpenGL
verifier report that breakdown.

### Draw orders and incremental preparation

Preparation now builds a draw order and draws from it. `shade_slots` colours
each survivor and copies its record into its compacted slot, the sort orders
the slots, and `finish_order` lays the survivors out in draw order as 64-byte
`DrawnSplat`s: record, colour, key and splat index. `reproject_gather` then
projects them in sequence for the frame's camera and writes the indirect draw
arguments. A full preparation does all of that in one frame and draws the same
stream as before.

While the camera moves, the display prepares incrementally: every frame draws
the newest complete order reprojected, and the next order is built one step a
frame, five steps of comparable cost (culling, each half of the colour, each
half of the sort). An order built for the camera of one frame is drawn from
four to eight frames later; building culls against a frustum 15% wider, so
turning uncovers no gaps; the first frame the camera stands still is prepared
exactly. Only the orders'
counts are kept twice: the survivors in draw order are replaced as a build
completes, by which point nothing reads the old ones. The Incremental
Preparation property turns it off. Both verifiers check that frames drawing an
old order match the CPU path for that order projected with the new camera, and
that the completed order matches the CPU path for the new view.

Drawing reads the order in sequence because random reads over the whole scene
fell off a cliff on the RTX 4090: 0.9 ms spread over 110 MB, 22 ms over the
280 MB of Garden's records, whatever was done with them. The old gather escaped
it only because it read a buffer the size of the visible set, just written. For
the same reason `finish_order`, the one permutation per order, runs in passes
over 64 MiB ranges of slots.

GPU time per frame, median (largest) of the incremental frames:

| Scene | Full preparation | Incremental frame |
| --- | ---: | ---: |
| M3, real Garden, training camera 0, 2940x1905 | 28.4 ms | 7.9 (10.3) ms |
| M3, real Garden, training camera 40 | 26.5 ms | 7.4 (9.5) ms |
| M3, synthetic Garden-sized | 29.7 ms | 8.1 (10.6) ms |
| RTX 4090, synthetic Garden-sized | 5.9 ms | 1.2 (2.5) ms |

In RViz on the M3, with the camera moving every frame over the real Garden
scene at 2940x1905, the frame rate went from 10-13 fps to 15-18 fps.

Colours computed when an order is built lag with it, and change in a step
when the next order takes over, which shows on specular highlights. Colour
Refresh, under Incremental Preparation, evaluates the spherical harmonics of
the order being drawn again for the current camera: every frame, or half the
survivors a frame in turn (the default), so no colour is more than a frame
old. `recolour_order` walks the survivors by slot, in index order, so the
coefficients are read as stored, and writes each colour to its draw position,
at random but within a buffer the size of the visible set; `finish_order`
keeps each slot's splat and draw position for it. Keeping those costs the
completing frame a few milliseconds on the M3, so an order completed with the
refresh off skips them, and the refresh starts with the next order when it is
switched on mid-motion. Both verifiers check refreshed colours against the CPU
colour for the new camera on the old order, and fail with the refresh
disabled. GPU time per incremental frame, median (largest):

| Scene | Off | Half per frame | Every frame |
| --- | ---: | ---: | ---: |
| M3, real Garden, training camera 0, 2940x1905 | 7.8 (9.9) ms | 12.9 (18.3) ms | 17.6 (23.5) ms |
| M3, real Garden, training camera 0, close | 4.3 (6.4) ms | 7.0 (9.3) ms | 9.7 (11.9) ms |
| RTX 4090, synthetic Garden-sized | 1.2 (1.5) ms | 2.1 (2.6) ms | 3.0 (3.5) ms |

### Offscreen precision

A pixel of the Garden scene blends about sixty splats, and an 8-bit target
rounds after every one, dropping layers too faint to move the value by half a
step and clipping highlights above 1.0 before the splats in front dim them.
Against float accumulation, 8-bit blending left 1-17% of pixels more than two
steps off on the M3 and 10-40% on the RTX 4090; blending into half float agrees
to within 0.01%. The offscreen target is now half float by default (Precision,
under Offscreen Rendering). It cost nothing measurable on the M3, whose blending
stays in tile memory, and 0.3-2.7 ms a frame on the RTX 4090.

### Where the M3 frame goes

In RViz on the M3 with the camera moving, the GPU is 100% busy (its tiler too)
and RViz uses about 10% of a CPU, so the frame is GPU-bound. A Metal System
Trace puts each 55-60 ms frame at about 14 ms of vertex and tiling work for the
splat pass, which the render pass cannot start fragments before, and about
26 ms of fragment work, overlapping and contending with the next frame's
preparation. Preparation is no longer the bottleneck there; drawing 2.26
million quads is.

Measured headless on the same view, the splat pass costs 23 ms at 320x207 and
31 ms at 2940x1905: its cost is mostly geometry, not fill. Of that, the vertex
stage alone is 13 ms, bound by the four instanced invocations per splat rather
than by the 48 bytes each fetches (32 bytes measured the same). Pulling the
instance data by vertex index instead of instancing cut the vertex stage to
5 ms and the whole pass at 320x207 from 25 to 19 ms, pixel-identical, but at
2940x1905 it saved nothing, since the tiler and fragments bound the pass there;
it would also need a 6-index-per-splat index buffer (140 MiB for Garden) and a
draw outside Ogre's instancing, so it is not done. Minimum Screen Radius culls
only 1-2% of Garden's splats at this resolution, too few to matter. One large
triangle per splat instead of two was slower. What remains to trade is quality:
Sigma Radius shrinks every quad, Render Scale only the fill.

### Culling by contribution

Many splats of a trained scene are faint and a few pixels across, and each
costs the tiler and the blender as much as a bright one. `splatContribution()`
measures what a projected splat adds to the image, its opacity integrated over
its screen footprint, alpha 2 pi sigma1 sigma2, in fully opaque pixels, and the
Minimum Contribution property (under Visibility Culling, 0 by default) leaves
out splats below it. It is checked in the projection every frame, so it follows
the camera in incremental preparation too; a culled splat becomes an empty
quad, clipped before tiling. GPU preparation only.

On the M3, full Garden scene at 2940x1905, measured while another RViz shared
the GPU, so the times are relative (no culling: 35.9 and 32.9 ms):

| Minimum | Culled, camera 0 | Splat pass, camera 0 / 40 | PSNR, camera 0 / 40 | Pixels > 2 steps off, camera 0 |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 14% | 31.8 / 30.5 ms | 50.3 / 54.3 dB | 1.1% |
| 0.25 | 26% | 29.3 / 29.5 ms | 43.3 / 46.0 dB | 3.9% |
| 0.5 | 35% | 28.6 / 27.9 ms | 39.4 / 41.3 dB | 8.2% |
| 1 | 43% | 26.4 / 25.8 ms | 35.7 / 37.2 dB | 15.5% |
| 2 | 53% | 24.0 / 24.2 ms | 31.3 / 32.9 dB | 27.4% |

The pass shrinks far less than the splat count, and close views, whose cost is
large splats, hardly at all (camera 0 moved 3.5 units in: 52.5 ms to 49.7 ms at
1, still at 51 dB). The losses are thin detail and a slight darkening, about a
step on average at 1. Around 0.1-0.25 the error is no larger than 8-bit blending
used to make.

Still open:

- Degree-3 SH is uploaded as float32, about 1 GiB at this count. Half precision
  would halve it, and the M3's colour step, now half of each build, is bound by
  reading it, at the cost of exact agreement with the CPU colours.
- The OpenGL scatters rank lanes with subgroup ballots where the driver offers
  GL_KHR_shader_subgroup with 32-lane subgroups (compaction's scatter 0.28 to
  0.05 ms, the radix scatters 1.08 to 0.95 ms on the RTX 4090). Staging each
  tile in shared memory first was tried and was slower.
- Real frame rates in RViz on the RTX 4090 have not been measured since these
  changes; the numbers above for it are headless.
