# Known issues

## OpenGL data-texture addressing above 5,592,405 splats

Status: deferred.

This affects the CPU-prepared OpenGL fallback, but not Metal or the OpenGL 4.3
GPU-preparation path. The fallback's GLSL 1.20 vertex shader receives the splat
index as a `float`, multiplies it by the three texels in one record, and derives
the texture row and column from that value. A single-precision float represents
every integer only through 2^24, so the address is no longer exact above
5,592,405 splats. The full Garden scene has 5,834,784 splats and crosses that
boundary.

The display currently accepts the message and reports a warning. Some splats
can consequently read an adjacent record on the fallback. Metal uses integer
texture coordinates, while OpenGL 4.3 compute writes projected instances from
SSBOs; neither is affected.

Current workarounds:

- Use Metal or OpenGL 4.3 GPU preparation for a full scene.
- Publish fewer splats (Garden with stride 2 is below the limit).

Remaining fallback fix: carry an integer-safe texture coordinate rather than a
single large floating-point index. For GLSL 1.20 this can be a row and a column,
both of which remain small exact integers. The OpenGL 4.3 path already avoids
the issue by keeping the index integer and gathering from SSBOs.
