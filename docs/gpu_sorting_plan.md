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
with the license retained in `tools/THIRD_PARTY_NOTICES.md`.

The exact 32-bit radix reduces the measured 1.65-million-key CPU radix from
11.60 ms to about 5.77 ms. A 16-bit half-depth key needs only two passes and
reduces that to about 1.93 ms, but it changes ordering precision. It is therefore
an optional candidate, not the default, until deterministic image comparisons
show that equal/quantised depth ties do not produce visible blending artifacts.

These are sort-only numbers. They do not alter the earlier conclusion that the
whole GPU path must include culling, compaction, SH evaluation, gather, and an
indirect draw: the measured CPU cull/SH stage is still the dominant 60.93 ms.

### OpenGL

Ogre's GL3Plus backend implements `_dispatchCompute()` and shader-storage
buffers, so a GL 4.3 implementation can share the algorithm and buffer layout
with Metal. The current macOS/legacy path uses GLSL 1.20 and cannot run compute
shaders; it must retain the CPU implementation. GPU sorting is therefore a
capability-selected acceleration, not a replacement for the portable path.

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
