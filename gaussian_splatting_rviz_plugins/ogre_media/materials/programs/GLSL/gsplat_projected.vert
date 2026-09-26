#version 120

// Prepared once per visible splat by the GL 4.3 compute backend.
attribute vec4 uv0;    // NDC centre xyz, visible radius
attribute vec4 uv1;    // two scaled ellipse axes in NDC
attribute vec4 colour; // RGB and compensated opacity
attribute vec2 uv2;    // quad corner in [-1, 1]

varying vec4 splat_colour;
varying vec2 splat_local_coord;

void main()
{
  gl_Position = vec4(
    uv0.xy + uv2.x * uv1.xy + uv2.y * uv1.zw,
    uv0.z, 1.0);
  splat_colour = colour;
  splat_local_coord = uv2 * uv0.w;
}
