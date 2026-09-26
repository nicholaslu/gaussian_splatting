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

On NVIDIA systems, CUB `DeviceRadixSort::SortPairs` is the appropriate optional
backend. CUDA/OpenGL buffer interop can keep its output on the GPU. It cannot
serve Metal or Apple devices, and adding it before the renderer-independent
buffer contract exists would duplicate integration work. Add it after the
Metal/GL compute interface is stable.

The reference 3DGS rasterizer goes further: it emits tile/depth keys, sorts
key/value pairs with CUB, identifies each tile's sorted range, and blends tiles
on the GPU. That architecture is a useful later target, but it is not a drop-in
replacement for this plugin's single global alpha-blended Ogre draw.

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

Still open:

- Degree-3 SH is uploaded as float32, about 1 GiB at this count. Uploads now
  reuse the buffers in place, but the precision question is undecided: half
  precision would halve it at the cost of exact agreement with the CPU colours.
- Gather, SH and projection, 26 ms at Garden scale, is the largest stage.
  Frame pacing is the next target: the sort could be spread over several frames
  in slices, with the gather re-projecting the last completed order every
  frame. That needs a guard band on the culling, since a stale cull would
  otherwise drop splats entering the view.
