# OpenGL performance with a moving view on the full Garden scene

## Background and scope

This note collects the performance behaviour of the GaussianSplatting RViz plugin on the full Garden dataset, and makes a first comparison between the Metal and OpenGL 4.3 GPU preparation paths.

The focus of this round is not the static frame rate but sustained performance and interaction latency while the camera moves. The scene has about **5,834,784 splats**; typical views have about **900,000 to 1,000,000 splats visible**. The remote OpenGL test machine uses an NVIDIA RTX 4090.

What follows is split into observed behaviour, directions that have been largely ruled out, and preliminary conclusions drawn from the code. Anything not yet confirmed by per-stage GPU timing is marked as inference.

## Observations

### Static and moving views differ sharply

- Once the camera stops, RViz returns to about 60 FPS.
- While the camera keeps moving, the frame rate drops markedly and interaction becomes sluggish.
- After the publisher is stopped, moving the view over the already loaded scene does not noticeably recover the frame rate.

So the bottleneck is not continuous ROS publishing, deserialisation or repeated upload, but per-view GPU work that the loaded scene triggers whenever the view changes.

### OpenGL GPU preparation is enabled

- The OpenGL 4.3 GPU preparation path is enabled and running.
- The CPU needs only about **0.06–0.075 ms** to submit one preparation to the GPU.
- The RViz process uses about **22%** CPU; the publisher is essentially idle at steady state.
- While the view moves, NVIDIA GPU utilisation stays at **96–100%**.

The low frame rate is therefore not the old CPU sort problem coming back; GPU preparation, or the way it is scheduled, has become the main limit.

### Lowering the output resolution does not remove the saturation

Lowering the Offscreen Render Scale from `1.0` to `0.25` should cut the pixel count to 1/16, yet the GPU stays at about 98–100% utilisation while moving.

This largely rules out final splat rasterisation and fragment fill as the sole cause. Fill and blending still cost something, but they cannot explain a GPU that stays saturated after such a large drop in resolution.

### Fewer visible splats still saturate the GPU

With Minimum Screen Radius set to `8`, the visible count falls from about 1 million to **515,174**, but the GPU is still close to fully loaded while moving.

That setting only reduces the number of survivors after culling; it does not spare the first stage from scanning all 5,834,784 inputs. The result points at culling, compaction and the other preparation stages that run over the whole input, not only at drawing the survivors.

### High GPU utilisation, but low power and memory utilisation

During the tests GPU utilisation was close to 100%, but power draw was about 95 W and memory utilisation about 0–2%. These figures cannot be mapped directly to the time of any particular kernel, but they look more like a workload limited by low occupancy, synchronisation or scheduling, or long serial instruction chains, than one saturating memory bandwidth or fragment throughput.

## The current GPU preparation pipeline

At a high level Metal and OpenGL run the same algorithm:

1. Cull every splat and generate its depth key.
2. Count and scan the survivors.
3. Compact the survivors into contiguous key/index buffers.
4. Run four passes of an 8-bit stable LSD radix sort.
5. Project, evaluate SH and gather according to the sorted order.
6. Write the indirect draw arguments and draw.

So OpenGL is not slower because it still takes the full CPU sort path, nor because Metal uses a different high-level algorithm.

## Key execution differences between Metal and OpenGL

### SIMD lane ranking

Metal's compact scatter and radix scatter compute lane ranks with the native `simd_ballot`.

The current OpenGL 4.3 GLSL path uses no subgroup ballot; it computes a stable rank within a 32-lane tile by looping over the other lanes and comparing. That costs more instructions and serial dependencies, and on NVIDIA GPUs is likely to be much less efficient than native warp/subgroup operations.

### Dispatch and synchronisation

Metal encodes the stages of a preparation on a single Metal command timeline, with fairly fine-grained buffer barriers.

The OpenGL path has to repeatedly:

- switch compute programs;
- rebind SSBOs;
- issue compute dispatches;
- execute `glMemoryBarrier`.

One preparation contains a dozen or more dispatches, each with its own stage synchronisation. Even if each kernel is short, the global barriers, driver scheduling and gaps between kernels can add up to significant overhead.

### Queuing policy for view updates

Neither Metal nor OpenGL currently has an explicit mechanism to coalesce or throttle view updates, keeping only the latest view while the previous preparation is still running.

Dragging the camera produces a continuous stream of view updates. Submitting one OpenGL preparation takes the CPU only about 0.07 ms, so the CPU can keep submitting the full pipeline far faster than the GPU consumes it. If intermediate views cannot be dropped in time, work piles up in the GPU queue, which shows up as:

- a low frame rate while moving;
- growing latency from input to screen;
- a wait after the camera stops while the queue drains.

This is a structural risk shared by both backends, not specific to OpenGL.

## Preliminary conclusions

The evidence so far supports the following:

1. **The main bottleneck is the GPU preparation triggered by camera changes, not the publisher or the CPU sort.**
2. **The problem is not simply fragment fill.** The GPU staying saturated at Render Scale 0.25 shows that preparation weighs heavily.
3. **The full scan is the base cost that grows with the scene.** Even with only about 510,000 survivors, the first stage still processes all of the roughly 5.83 million inputs.
4. **The OpenGL implementation is probably less efficient than Metal's.** The most suspicious differences are the hand-emulated 32-lane stable rank and the large number of dispatches and global barriers.
5. **The lack of view-update coalescing may amplify every GPU cost.** Even if a single preparation is only slightly slower than the target frame time, continuous dragging can submit far more work than the GPU can consume.
6. **Metal does not avoid the full scan algorithmically.** It may do better thanks to native SIMD primitives and the command-buffer model, but it can degrade too at the same 5.83 million scale with continuous view updates.

Overall GPU utilisation alone cannot tell whether compaction, the radix sort, the gather or waiting on barriers takes the largest share. That needs real per-stage GPU timing.

## Suggested order of further investigation

### 1. Add OpenGL GPU timestamp queries

Record the GPU time of each of these stages:

- cull/key generation;
- compact count/scan/scatter;
- each radix pass's histogram/scan/scatter;
- projection/SH/gather;
- the final draw.

Also record the total time from the start of the first stage to the end of the final draw, so that barriers and gaps between dispatches are not missed by looking only at the kernels.

### 2. Add latest-view coalescing and an in-flight limit

If views B, C and D arrive while the GPU is still processing view A, the unsubmitted intermediate views should be droppable, with only the latest, D, processed once submission is possible again. The final view must still be processed exactly once the camera stops.

This is the most direct way to improve interaction, and it also separates the effect of a single slow preparation from that of a queue backed up by repeated submissions.

### 3. Add a native subgroup path to OpenGL

Detect at run time whether the NVIDIA GLSL implementation offers subgroup ballots, and replace the hand-written 32-lane loop in GLSL with a native warp/subgroup rank, keeping the current plain OpenGL 4.3 implementation as the compatible fallback.

Compare the per-stage times of the compact scatter and the radix scatter, not just the final FPS.

### 4. Then reconsider hierarchical culling and CUDA/CUB

If the full scan in cull/key generation still dominates, consider a hierarchical bounding structure or tiled visibility, so that not every view scans every splat.

CUDA/CUB can give NVIDIA a mature radix primitive, but it mainly improves the sort stage; on its own it solves neither the full-scan culling nor repeated view submission. Its priority should be decided after the timestamp results confirm the sort's share.

## Boundary with an unrelated problem

The OpenGL floating-point texture indexing precision problem above 5,592,405 splats belongs to the CPU-prepared GLSL 1.20 fallback. The current OpenGL 4.3 GPU preparation indexes integer SSBOs and is not affected; it is not the cause of the low frame rate with a moving view.
