#version 120

attribute vec4 vertex;
attribute vec4 colour;
attribute vec3 uv0;
attribute vec3 uv1;

uniform mat4 projmatrix;
uniform mat4 viewmatrix;
uniform float fovy;
uniform vec4 vpsize;

varying vec4 splat_colour;
varying vec3 splat_conic;
varying vec2 splat_radius;

vec3 computeCov2D(vec3 position, vec3 diag, vec3 upper, out float compensation)
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

  mat3 Vrk = mat3(
    diag.x, upper.x, upper.y,
    upper.x, diag.y, upper.z,
    upper.y, upper.z, diag.z
  );

  mat3 cov = transpose(T) * Vrk * T;

  float det_orig = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;
  float det_blur = cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1];
  compensation = sqrt(max(det_orig / det_blur, 0.0));

  return vec3(cov[0][0], cov[0][1], cov[1][1]);
}

void main()
{
  vec3 position = vertex.xyz;
  vec4 p_hom = projmatrix * vec4(position, 1.0);
  vec3 p_proj = p_hom.xyz / p_hom.w;

  splat_colour = colour;

  float compensation;
  vec3 cov = computeCov2D(position, uv0, uv1, compensation);
  splat_colour.a *= compensation;

  if (splat_colour.a < 0.00392156862) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }

  float det = cov.x * cov.z - cov.y * cov.y;
  float det_inv = 1.0 / det;
  splat_conic = vec3(cov.z, -cov.y, cov.x) * det_inv;

  float mid = 0.5 * (cov.x + cov.z);
  float root = sqrt(max(0.1, mid * mid - det));
  float lambda1 = mid + root;
  float lambda2 = mid - root;
  float radius_px = ceil(3.0 * sqrt(max(lambda1, lambda2)));

  splat_radius = vec2(radius_px);
  gl_PointSize = 2.0 * radius_px;
  gl_Position = vec4(p_proj.xy, p_proj.z, 1.0);
}
