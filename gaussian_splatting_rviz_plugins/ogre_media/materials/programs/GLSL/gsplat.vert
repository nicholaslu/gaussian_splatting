#version 120

// The per-instance data: which splat this instance draws, and the colour its
// spherical harmonics give for this view. Everything else is read from
// splat_data, so a depth sort moves 8 bytes per splat, not a whole record.
attribute float uv0;    // splat index
attribute vec3 colour;  // view-dependent RGB, unclamped above 1
attribute vec2 uv2;     // quad corner in [-1, 1]

uniform mat4 projmatrix;
uniform mat4 viewmatrix;
uniform float fovy;
uniform vec4 vpsize;
uniform float eps2d;
uniform float antialiased;

// Per-splat records, four consecutive RGBA32F texels each. splat_data_size is
// (width, height, 1/width, 1/height): GLSL 120 has no integer texel fetch, so
// the reciprocals are needed to land on a texel centre.
uniform sampler2D splat_data;
uniform vec4 splat_data_size;

// Extent of the rasterised quad, in standard deviations. Fill rate scales with
// the square of this value, so lowering it is the cheapest accuracy/speed
// trade available; 3.0 matches the reference rasteriser.
uniform float sigma_radius;

varying vec4 splat_colour;
varying vec2 splat_local_coord;

// The texture's width is a whole number of splats, so a splat never straddles
// a row. The 0.5 lands on the texel centre, which is what makes an unfiltered
// sample equivalent to a fetch.
vec4 readTexel(float texel)
{
  float column = mod(texel, splat_data_size.x);
  float row = floor(texel * splat_data_size.z);
  return texture2DLod(
    splat_data,
    vec2((column + 0.5) * splat_data_size.z, (row + 0.5) * splat_data_size.w),
    0.0);
}

// Sigma = M * M^T with M = R * diag(s), i.e. column i of the rotation matrix
// scaled by s[i]. Matches build_scaling_rotation() in the reference
// implementation. mat3() takes columns.
mat3 computeCov3D(vec3 s, vec4 q)
{
  float x = q.x;
  float y = q.y;
  float z = q.z;
  float w = q.w;

  mat3 m = mat3(
    (1.0 - 2.0 * (y * y + z * z)) * s.x,
    (2.0 * (x * y + w * z)) * s.x,
    (2.0 * (x * z - w * y)) * s.x,

    (2.0 * (x * y - w * z)) * s.y,
    (1.0 - 2.0 * (x * x + z * z)) * s.y,
    (2.0 * (y * z + w * x)) * s.y,

    (2.0 * (x * z + w * y)) * s.z,
    (2.0 * (y * z - w * x)) * s.z,
    (1.0 - 2.0 * (x * x + y * y)) * s.z);

  return m * transpose(m);
}

vec3 computeCov2D(vec3 position, mat3 Vrk, out float compensation)
{
  float tan_fovy = tan(fovy * 0.5);
  float tan_fovx = tan_fovy * vpsize.x / vpsize.y;
  float focal_y = vpsize.y / (2.0 * tan_fovy);
  float focal_x = vpsize.x / (2.0 * tan_fovx);

  vec4 t = viewmatrix * vec4(position, 1.0);

  float limx = 1.3 * tan_fovx;
  float limy = 1.3 * tan_fovy;
  float txtz = t.x / t.z;
  float tytz = t.y / t.z;
  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  mat3 J = mat3(
    focal_x / t.z, 0.0, -(focal_x * t.x) / (t.z * t.z),
    0.0, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0.0, 0.0, 0.0
  );

  mat3 W = transpose(mat3(viewmatrix));
  mat3 T = W * J;

  mat3 cov = transpose(T) * Vrk * T;

  float det_orig = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
  cov[0][0] += eps2d;
  cov[1][1] += eps2d;
  float det_blur = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
  compensation = sqrt(max(det_orig / det_blur, 0.0));

  return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
  float texel = uv0 * 3.0;
  vec4 a = readTexel(texel);
  vec4 b = readTexel(texel + 1.0);
  vec4 c = readTexel(texel + 2.0);

  vec3 position = a.xyz;
  vec3 scale = b.xyz;
  vec4 quat = c;

  vec4 p_hom = projmatrix * vec4(position, 1.0);

  // Behind the camera the manual perspective divide below would wrap the splat
  // onto the opposite side of the screen, so discard it here instead.
  if (p_hom.w <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    splat_colour = vec4(0.0);
    splat_local_coord = vec2(0.0);
    return;
  }
  vec3 p_proj = p_hom.xyz / p_hom.w;

  splat_colour = vec4(colour, a.w);

  float compensation;
  vec3 cov = computeCov2D(position, computeCov3D(scale, quat), compensation);

  // The compensation factor is only correct for opacities that were optimised
  // with it applied, so it follows the message's rasterize_mode.
  splat_colour.a *= mix(1.0, compensation, antialiased);

  const float alpha_cutoff = 0.00392156862;
  if (splat_colour.a < alpha_cutoff) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    splat_local_coord = vec2(0.0);
    return;
  }

  // alpha = opacity * exp(-r^2 / 2). Pixels beyond this radius would be
  // discarded by the fragment shader, so do not rasterise them in the first
  // place. This is exact with respect to the shader's alpha cutoff and can
  // shrink low-opacity splats substantially below the configured sigma cap.
  float visible_radius = min(
    sigma_radius,
    sqrt(2.0 * log(splat_colour.a / alpha_cutoff)));

  // The eigen decomposition as in ../Metal/gsplat.metal, whose comments say
  // why its formulas avoid cancellation.
  float mid = 0.5 * (cov.x + cov.z);
  float half_difference = 0.5 * (cov.x - cov.z);
  float root = sqrt(half_difference * half_difference + cov.y * cov.y);
  float lambda1 = max(mid + root, 0.01);
  float lambda2 = max(mid - root, 0.01);

  vec2 axis1 = half_difference >= 0.0 ?
    vec2(half_difference + root, cov.y) : vec2(cov.y, root - half_difference);
  float axis_length = length(axis1);
  axis1 = axis_length > 0.0 ? axis1 / axis_length : vec2(1.0, 0.0);
  vec2 axis2 = vec2(-axis1.y, axis1.x);

  vec2 pixel_offset =
    uv2.x * axis1 * (visible_radius * sqrt(lambda1)) +
    uv2.y * axis2 * (visible_radius * sqrt(lambda2));
  vec2 ndc_offset = pixel_offset * 2.0 / vpsize.xy;

  splat_local_coord = uv2 * visible_radius;
  gl_Position = vec4(p_proj.xy + ndc_offset, p_proj.z, 1.0);
}
