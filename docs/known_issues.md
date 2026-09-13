# Known issues

## OpenGL data-texture addressing above 5,592,405 splats

Status: deferred.

This affects the legacy OpenGL path, but not Metal. The GLSL 1.20 vertex
shader receives the splat index as a `float`, multiplies it by the three texels
in one record, and derives the texture row and column from that value. A
single-precision float represents every integer only through 2^24, so the
address is no longer exact above 5,592,405 splats. The full Garden scene has
5,834,784 splats and crosses that boundary.

The display currently accepts the message and reports a warning. Some splats
can consequently read an adjacent record on OpenGL. Metal uses integer texture
coordinates and is unaffected.

Current workarounds:

- Use Metal for a full scene.
- Publish fewer splats (Garden with stride 2 is below the limit).

Planned fix: carry an integer-safe texture coordinate rather than a single
large floating-point index. For GLSL 1.20 this can be a row and a column, both
of which remain small exact integers. A future GL 4.3 path can instead use an
integer index and `texelFetch`. The solution should be integrated with the GPU
sorting work so the GPU-generated draw order does not reintroduce the same
float index.

