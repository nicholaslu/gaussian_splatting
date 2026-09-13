#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gaussian_splatting_rviz_plugins
{

// The per-view preparation contract. The display's CPU path runs prepareSplat()
// below, the Metal kernels in gsplat_prepare.metal mirror these layouts and
// this arithmetic, and test/verify_gpu_preparation.mm runs both on the same
// ViewParameters. A CPU/GPU disagreement can therefore only come from the
// kernels, which is exactly what the verifier checks.

// One splat as the vertex shader reads it from the data texture, laid out as
// whole texels so a row of the texture is a whole number of splats. Colour is
// absent because it is view dependent; it arrives per instance instead.
struct SplatRecord
{
  float position[3];
  float opacity;
  float scale[3];
  float pad0;
  float quat[4];
};
static_assert(sizeof(SplatRecord) == 12 * sizeof(float), "three texels per splat");

// What each drawn instance carries: which splat, and the colour its
// spherical harmonics give for this view. Sixteen bytes, against the 56 the
// whole record used to cost per camera move.
//
// Float rather than a packed byte colour: the reference rasteriser keeps
// max(sh, 0) with no upper bound and only clamps after blending, and on this
// scene 7.5% of opaque splats have a channel above 1.0, reaching 3.5. Those
// are the specular highlights, so clamping them per splat flattens exactly
// the surfaces the higher order coefficients exist to reproduce.
struct DrawInstance
{
  float index;
  float colour[3];
};
static_assert(sizeof(DrawInstance) == 16, "index plus an unclamped colour");

// ViewParameters::flags.
constexpr std::uint32_t kViewCulling = 1u;
constexpr std::uint32_t kViewPerspective = 2u;

// The far plane's index in Ogre's FrustumPlane order, which frustum_planes
// follows. A far clip distance of zero means an infinite frustum, and
// Frustum::isVisible() skips that plane.
constexpr int kFrustumPlaneFar = 1;

// Everything per-splat preparation needs from the camera and the display, as
// plain values. Matrices are row-major, so a transform is the same row dot
// products Ogre performs; frustum planes are (normal, d). Mirrored by
// ViewParameters in gsplat_prepare.metal.
struct alignas(16) ViewParameters
{
  float world_rows[16];
  float view_rows[16];
  float frustum_planes[24];
  float local_camera[4];
  float local_sort_direction[4];
  float world_scale;
  float focal_y;
  float near_clip;
  float far_clip;
  float ortho_scale;
  float sigma_radius;
  float min_screen_radius;
  float eps2d;
  std::uint32_t count;
  std::uint32_t viewport_height;
  std::uint32_t sh_coefficients;
  std::uint32_t flags;
};
static_assert(sizeof(ViewParameters) == 304, "Metal parameter layout changed");

// Spherical harmonics basis constants, as used by the 3DGS reference
// implementation. Degree 0 alone gives rgb = sh_dc * kSHC0 + 0.5; the rest add
// the view dependence that a specular surface is almost entirely made of.
constexpr float kSHC0 = 0.28209479177387814f;
constexpr std::uint8_t kMaxShDegree = 3;
constexpr float kSHC1 = 0.4886025119029199f;
constexpr float kSHC2[5] = {
  1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
  -1.0925484305920792f, 0.5462742152960396f,
};
constexpr float kSHC3[7] = {
  -0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
  0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
  -0.5900435899266435f,
};

// Maps a float onto a uint32 whose unsigned order matches the float's ordering,
// so depths can be sorted by their bits: flip the sign bit for positives, and
// invert everything for negatives, whose magnitude ordering runs backwards.
// This also gives NaN a defined place, which the old comparator did not.
inline std::uint32_t depthKey(float depth)
{
  std::uint32_t bits;
  std::memcpy(&bits, &depth, sizeof(bits));
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Evaluates the spherical harmonics for one splat along a unit view direction.
// This is eval_sh() from the 3DGS reference implementation, degree for degree;
// `rest` is laid out coefficient major, matching sh_rest in GaussianSplats.msg.
// Like the reference it clamps below at zero and not above, leaving highlights
// brighter than white to be resolved by blending.
inline void evaluateColour(
  const float * dc, const float * rest, std::size_t coefficients, const float view[3],
  float * colour)
{
  float channel[3];
  for (int c = 0; c < 3; ++c) {
    channel[c] = kSHC0 * dc[c] + 0.5f;
  }

  const float x = view[0];
  const float y = view[1];
  const float z = view[2];

  if (coefficients >= 3) {
    for (int c = 0; c < 3; ++c) {
      channel[c] += -kSHC1 * y * rest[c] + kSHC1 * z * rest[3 + c] - kSHC1 * x * rest[6 + c];
    }
  }
  if (coefficients >= 8) {
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    for (int c = 0; c < 3; ++c) {
      channel[c] +=
        kSHC2[0] * xy * rest[9 + c] +
        kSHC2[1] * yz * rest[12 + c] +
        kSHC2[2] * (2.0f * zz - xx - yy) * rest[15 + c] +
        kSHC2[3] * xz * rest[18 + c] +
        kSHC2[4] * (xx - yy) * rest[21 + c];
    }
  }
  if (coefficients >= 15) {
    const float xx = x * x, yy = y * y, zz = z * z, xy = x * y;
    for (int c = 0; c < 3; ++c) {
      channel[c] +=
        kSHC3[0] * y * (3.0f * xx - yy) * rest[24 + c] +
        kSHC3[1] * xy * z * rest[27 + c] +
        kSHC3[2] * y * (4.0f * zz - xx - yy) * rest[30 + c] +
        kSHC3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * rest[33 + c] +
        kSHC3[4] * x * (4.0f * zz - xx - yy) * rest[36 + c] +
        kSHC3[5] * z * (xx - yy) * rest[39 + c] +
        kSHC3[6] * x * (xx - 3.0f * yy) * rest[42 + c];
    }
  }

  for (int c = 0; c < 3; ++c) {
    colour[c] = std::max(0.0f, channel[c]);
  }
}

// Ogre's VectorBase::dotProduct(), statement for statement. The form matters,
// not only the value: under the default floating-point contraction each `+=`
// becomes a fused multiply-add, while a single a*x + b*y + c*z expression fuses
// differently. Written that way, 10% of depth keys and 5% of colours stopped
// being bit-identical to the Ogre-based loop this replaced. Kept in Ogre's
// form, the CPU path is exactly what it was, and the Metal dot() matches it.
inline float dotProduct(float ax, float ay, float az, float bx, float by, float bz)
{
  float result = 0.0f;
  result += ax * bx;
  result += ay * by;
  result += az * bz;
  return result;
}

// Culls one splat against the view and, if it survives, writes its colour and
// depth key; `sh_rest` points at this splat's coefficients, or is null at
// degree 0. This is the body of the CPU path's cull loop and the reference the
// GPU kernels are checked against. The arithmetic is Frustum::isVisible(Sphere),
// Plane::getDistance(), Vector3::normalise() and Matrix4 * Vector3 written out
// in Ogre's own evaluation order. Matrix4 * Vector3 also multiplies by 1/w,
// which is exactly 1 for the affine node and view transforms passed in here.
inline bool prepareSplat(
  const ViewParameters & p, const SplatRecord & record,
  const float * sh_dc, const float * sh_rest, float * colour, std::uint32_t & key)
{
  const float alpha_cutoff = 1.0f / 255.0f;
  if (record.opacity < alpha_cutoff) {
    return false;
  }
  const bool culling_enabled = (p.flags & kViewCulling) != 0u;
  const bool perspective = (p.flags & kViewPerspective) != 0u;

  const float x = record.position[0];
  const float y = record.position[1];
  const float z = record.position[2];
  const float visible_radius = std::min(
    p.sigma_radius, std::sqrt(2.0f * std::log(record.opacity / alpha_cutoff)));
  const float max_sigma = std::max(record.scale[0], std::max(record.scale[1], record.scale[2]));
  const float world_radius = visible_radius * max_sigma * p.world_scale;
  const float wx =
    p.world_rows[0] * x + p.world_rows[1] * y + p.world_rows[2] * z + p.world_rows[3];
  const float wy =
    p.world_rows[4] * x + p.world_rows[5] * y + p.world_rows[6] * z + p.world_rows[7];
  const float wz = p.world_rows[8] * x + p.world_rows[9] * y + p.world_rows[10] * z +
    p.world_rows[11];

  if (culling_enabled) {
    for (int plane = 0; plane < 6; ++plane) {
      if (plane == kFrustumPlaneFar && p.far_clip == 0.0f) {
        continue;
      }
      const float * n = &p.frustum_planes[plane * 4];
      if (dotProduct(n[0], n[1], n[2], wx, wy, wz) + n[3] < -world_radius) {
        return false;
      }
    }
  }

  if (culling_enabled && p.min_screen_radius > 0.0f && p.viewport_height > 0u) {
    float screen_radius = 0.0f;
    if (perspective) {
      const float depth =
        -(p.view_rows[8] * wx + p.view_rows[9] * wy + p.view_rows[10] * wz + p.view_rows[11]);
      if (depth <= 0.0f) {
        return false;
      }
      screen_radius = p.focal_y * world_radius / std::max(depth - world_radius, p.near_clip);
    } else {
      screen_radius = p.ortho_scale * world_radius;
    }

    // eps2d adds this minimum variance in screen space before rasterisation.
    const float blur_radius = visible_radius * std::sqrt(std::max(p.eps2d, 0.0f));
    screen_radius = std::sqrt(screen_radius * screen_radius + blur_radius * blur_radius);
    if (screen_radius < p.min_screen_radius) {
      return false;
    }
  }

  // Vector3::normalise(), guard included: a splat exactly at the camera keeps
  // a zero direction, where an unguarded normalise would make it NaN.
  float view[3] = {x - p.local_camera[0], y - p.local_camera[1], z - p.local_camera[2]};
  const float length =
    std::sqrt(dotProduct(view[0], view[1], view[2], view[0], view[1], view[2]));
  if (length > 0.0f) {
    const float inverse = 1.0f / length;
    view[0] *= inverse;
    view[1] *= inverse;
    view[2] *= inverse;
  }
  evaluateColour(sh_dc, sh_rest, p.sh_coefficients, view, colour);

  key = depthKey(dotProduct(
      x, y, z, p.local_sort_direction[0], p.local_sort_direction[1], p.local_sort_direction[2]));
  return true;
}

// The per-view projection the splat vertex program used to derive from Ogre's
// auto parameters, once per vertex: worldview_matrix, worldviewproj_matrix with
// the render system's depth range, viewport_size of the viewport being
// rasterised, and fov. GPU preparation computes it once per splat in
// gather_instances instead, and the vertex program only places the corners.
// Mirrored by ProjectionParameters in gsplat_prepare.metal.
struct alignas(16) ProjectionParameters
{
  float worldview_rows[16];
  float worldviewproj_rows[16];
  float viewport_size[4];
  float fovy;
  float eps2d;
  float antialiased;
  float sigma_radius;
};
static_assert(sizeof(ProjectionParameters) == 160, "Metal projection layout changed");

// One splat projected to the screen: everything gsplat_projected_vp needs to
// place and shade its quad. A splat behind the camera, or faded below the alpha
// cutoff, sits outside the clip volume with no extent and rasterises nothing.
struct ProjectedInstance
{
  float centre[3];       // normalised device coordinates, depth included
  float visible_radius;  // in standard deviations
  float axes[4];         // the ellipse's two scaled axes in NDC: x1, y1, x2, y2
  float colour[4];       // rgb, and opacity after the antialiasing compensation
};
static_assert(sizeof(ProjectedInstance) == 48, "four float quadruples");

// gsplat_vp from gsplat.metal for one splat, less the corner displacement: the
// reference the projection in gather_instances is checked against.
inline void projectSplat(
  const ProjectionParameters & q, const SplatRecord & splat, const float colour[3],
  ProjectedInstance & out)
{
  out = ProjectedInstance{{0.0f, 0.0f, 2.0f}, 0.0f, {0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f}};

  const float x = splat.position[0];
  const float y = splat.position[1];
  const float z = splat.position[2];
  const auto row = [x, y, z](const float * rows, int r) {
      return rows[r * 4] * x + rows[r * 4 + 1] * y + rows[r * 4 + 2] * z + rows[r * 4 + 3];
    };

  // Behind the camera the perspective divide would wrap the splat onto the
  // opposite side of the screen.
  const float clip_w = row(q.worldviewproj_rows, 3);
  if (clip_w <= 0.0f) {
    return;
  }

  // Sigma = M M^T with M = R diag(s).
  const float qx = splat.quat[0];
  const float qy = splat.quat[1];
  const float qz = splat.quat[2];
  const float qw = splat.quat[3];
  const float rotation[3][3] = {
    {1.0f - 2.0f * (qy * qy + qz * qz), 2.0f * (qx * qy - qw * qz),
      2.0f * (qx * qz + qw * qy)},
    {2.0f * (qx * qy + qw * qz), 1.0f - 2.0f * (qx * qx + qz * qz),
      2.0f * (qy * qz - qw * qx)},
    {2.0f * (qx * qz - qw * qy), 2.0f * (qy * qz + qw * qx),
      1.0f - 2.0f * (qx * qx + qy * qy)},
  };
  float m[3][3];
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      m[r][c] = rotation[r][c] * splat.scale[c];
    }
  }
  float sigma[3][3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      sigma[i][j] = m[i][0] * m[j][0] + m[i][1] * m[j][1] + m[i][2] * m[j][2];
    }
  }

  // EWA: the rows of J W, with J the Jacobian of the perspective projection at
  // the splat, clamped to 1.3x the field of view as in the reference, and W the
  // linear part of worldview.
  const float width = q.viewport_size[0];
  const float height = q.viewport_size[1];
  const float tan_fovy = std::tan(q.fovy * 0.5f);
  const float tan_fovx = tan_fovy * width / height;
  const float focal_y = height / (2.0f * tan_fovy);
  const float focal_x = width / (2.0f * tan_fovx);
  const float tz = row(q.worldview_rows, 2);
  const float limit_x = 1.3f * tan_fovx;
  const float limit_y = 1.3f * tan_fovy;
  const float tx = std::min(limit_x, std::max(-limit_x, row(q.worldview_rows, 0) / tz)) * tz;
  const float ty = std::min(limit_y, std::max(-limit_y, row(q.worldview_rows, 1) / tz)) * tz;
  const float j0x = focal_x / tz;
  const float j0z = -(focal_x * tx) / (tz * tz);
  const float j1y = focal_y / tz;
  const float j1z = -(focal_y * ty) / (tz * tz);
  float t0[3];
  float t1[3];
  for (int c = 0; c < 3; ++c) {
    t0[c] = j0x * q.worldview_rows[c] + j0z * q.worldview_rows[8 + c];
    t1[c] = j1y * q.worldview_rows[4 + c] + j1z * q.worldview_rows[8 + c];
  }
  const auto quadratic = [&sigma](const float * u, const float * v) {
      float result = 0.0f;
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          result += u[i] * sigma[i][j] * v[j];
        }
      }
      return result;
    };
  float a = quadratic(t0, t0);
  const float b = quadratic(t0, t1);
  float c = quadratic(t1, t1);

  const float det_original = a * c - b * b;
  a += q.eps2d;
  c += q.eps2d;
  const float det_blurred = a * c - b * b;
  const float compensation = std::sqrt(std::max(det_original / det_blurred, 0.0f));
  // The compensation is only right for opacities optimised with it applied.
  const float alpha = splat.opacity * (1.0f + (compensation - 1.0f) * q.antialiased);
  const float alpha_cutoff = 1.0f / 255.0f;
  if (alpha < alpha_cutoff) {
    return;
  }

  const float visible_radius =
    std::min(q.sigma_radius, std::sqrt(2.0f * std::log(alpha / alpha_cutoff)));
  // The eigen decomposition as in gsplat_vp, whose comments say why its
  // formulas avoid cancellation.
  const float mid = 0.5f * (a + c);
  const float half_difference = 0.5f * (a - c);
  const float root = std::sqrt(half_difference * half_difference + b * b);
  const float lambda1 = std::max(mid + root, 0.01f);
  const float lambda2 = std::max(mid - root, 0.01f);
  float axis_x = half_difference >= 0.0f ? half_difference + root : b;
  float axis_y = half_difference >= 0.0f ? b : root - half_difference;
  const float axis_length = std::sqrt(axis_x * axis_x + axis_y * axis_y);
  if (axis_length > 0.0f) {
    axis_x /= axis_length;
    axis_y /= axis_length;
  } else {
    axis_x = 1.0f;
    axis_y = 0.0f;
  }
  const float radius1 = visible_radius * std::sqrt(lambda1);
  const float radius2 = visible_radius * std::sqrt(lambda2);

  out.centre[0] = row(q.worldviewproj_rows, 0) / clip_w;
  out.centre[1] = row(q.worldviewproj_rows, 1) / clip_w;
  out.centre[2] = row(q.worldviewproj_rows, 2) / clip_w;
  out.visible_radius = visible_radius;
  out.axes[0] = axis_x * radius1 * 2.0f / width;
  out.axes[1] = axis_y * radius1 * 2.0f / height;
  out.axes[2] = -axis_y * radius2 * 2.0f / width;
  out.axes[3] = axis_x * radius2 * 2.0f / height;
  out.colour[0] = colour[0];
  out.colour[1] = colour[1];
  out.colour[2] = colour[2];
  out.colour[3] = alpha;
}

}  // namespace gaussian_splatting_rviz_plugins
