// A trained 3DGS scene and one of its training cameras, for the rasteriser
// benchmarks: the PLY a 3DGS training run writes, converted as
// tools/publish_gaussian_ply.py converts it for the message, and the
// cameras.json written next to it, turned into the ViewParameters and
// ProjectionParameters the display would build for that camera.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"
#include "preparation_test_scenes.hpp"

namespace gaussian_splatting_rviz_plugins
{
namespace verification
{

// Reads a binary little-endian 3DGS PLY: log scales, logit opacities, (w, x, y,
// z) quaternions and channel-major f_rest, into linear scales, opacities,
// normalised (x, y, z, w) quaternions and coefficient-major SH, the layout of
// GaussianSplats.msg.
inline bool loadPly(const std::string & path, Scene & scene, std::string & error)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    error = "cannot open " + path;
    return false;
  }
  std::string line;
  std::vector<std::string> properties;
  std::uint64_t vertices = 0;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    std::istringstream words(line);
    std::string word;
    words >> word;
    if (word == "format" && line.find("binary_little_endian") == std::string::npos) {
      error = "only binary little-endian PLY is supported";
      return false;
    } else if (word == "element") {
      std::string name;
      words >> name >> vertices;
    } else if (word == "property") {
      std::string type, name;
      words >> type >> name;
      if (type != "float") {
        error = "property " + name + " is not float";
        return false;
      }
      properties.push_back(name);
    } else if (word == "end_header") {
      break;
    }
  }
  const auto column = [&properties](const std::string & name) {
      for (std::size_t i = 0; i < properties.size(); ++i) {
        if (properties[i] == name) {
          return int(i);
        }
      }
      return -1;
    };
  std::size_t rest_count = 0;
  while (column("f_rest_" + std::to_string(rest_count)) >= 0) {
    ++rest_count;
  }
  const std::size_t coefficients = rest_count / 3;
  const int x = column("x"), opacity = column("opacity"), scale0 = column("scale_0"),
    rot0 = column("rot_0"), dc0 = column("f_dc_0"), rest0 = column("f_rest_0");
  if (x < 0 || opacity < 0 || scale0 < 0 || rot0 < 0 || dc0 < 0 || vertices == 0 ||
    (coefficients > 0 && rest0 < 0))
  {
    error = "not a 3DGS PLY";
    return false;
  }

  scene.count = std::uint32_t(vertices);
  scene.coefficients = std::uint32_t(coefficients);
  scene.records.resize(vertices);
  scene.dc.resize(vertices * 3);
  scene.rest.resize(vertices * coefficients * 3);
  const std::size_t stride = properties.size();
  std::vector<float> row(stride * 65536);
  for (std::uint64_t begin = 0; begin < vertices; begin += 65536) {
    const std::uint64_t rows = std::min<std::uint64_t>(65536, vertices - begin);
    if (!file.read(reinterpret_cast<char *>(row.data()), std::streamsize(rows * stride * 4))) {
      error = "PLY ends early";
      return false;
    }
    for (std::uint64_t r = 0; r < rows; ++r) {
      const float * v = &row[r * stride];
      const std::uint64_t i = begin + r;
      SplatRecord & record = scene.records[i];
      record.position[0] = v[x];
      record.position[1] = v[x + 1];
      record.position[2] = v[x + 2];
      record.opacity = 1.0f / (1.0f + std::exp(-v[opacity]));
      for (int c = 0; c < 3; ++c) {
        record.scale[c] = std::exp(v[scale0 + c]);
      }
      record.pad0 = 0.0f;
      const float w = v[rot0], qx = v[rot0 + 1], qy = v[rot0 + 2], qz = v[rot0 + 3];
      const float norm = std::sqrt(w * w + qx * qx + qy * qy + qz * qz);
      const float inverse = norm > 0.0f ? 1.0f / norm : 0.0f;
      record.quat[0] = qx * inverse;
      record.quat[1] = qy * inverse;
      record.quat[2] = qz * inverse;
      record.quat[3] = norm > 0.0f ? w * inverse : 1.0f;
      for (int c = 0; c < 3; ++c) {
        scene.dc[i * 3 + c] = v[dc0 + c];
      }
      for (std::size_t k = 0; k < coefficients; ++k) {
        for (int c = 0; c < 3; ++c) {
          scene.rest[(i * coefficients + k) * 3 + c] = v[rest0 + c * coefficients + k];
        }
      }
    }
  }
  return true;
}

struct TrainingCamera
{
  double position[3];
  double rotation[3][3];  // camera to world; columns are right, down, forward
  double width, height, fx, fy;
};

// The `index`th camera of a 3DGS cameras.json.
inline bool loadCamera(
  const std::string & path, int index, TrainingCamera & camera, std::string & error)
{
  std::ifstream file(path);
  const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  std::size_t at = 0;
  for (int i = 0; i <= index; ++i) {
    at = text.find("\"id\"", at + 1);
    if (at == std::string::npos) {
      error = "cameras.json has no camera " + std::to_string(index);
      return false;
    }
  }
  const std::size_t end = text.find('}', at);
  const auto numbers = [&](const char * key, double * out, int count) {
      std::size_t p = text.find(key, at);
      if (p == std::string::npos || p > end) {
        return false;
      }
      p += std::strlen(key);
      for (int n = 0; n < count; ++n) {
        p = text.find_first_of("-0123456789", p);
        char * next = nullptr;
        out[n] = std::strtod(text.c_str() + p, &next);
        p = std::size_t(next - text.c_str());
      }
      return true;
    };
  if (!numbers("\"position\"", camera.position, 3) ||
    !numbers("\"rotation\"", &camera.rotation[0][0], 9) ||
    !numbers("\"width\"", &camera.width, 1) || !numbers("\"height\"", &camera.height, 1) ||
    !numbers("\"fx\"", &camera.fx, 1) || !numbers("\"fy\"", &camera.fy, 1))
  {
    error = "cameras.json camera " + std::to_string(index) + " lacks a field";
    return false;
  }
  return true;
}

// Moves the camera `distance` along its view direction, into the scene.
inline void moveForward(TrainingCamera & camera, double distance)
{
  for (int k = 0; k < 3; ++k) {
    camera.position[k] += camera.rotation[k][2] * distance;
  }
}

// The display's view of the scene from `camera`, rendered at `width` pixels
// wide with the camera's aspect and vertical field of view, the splat node at
// the origin. `zero_to_one_depth` picks Metal's clip depth range over OpenGL's.
struct CameraView
{
  ViewParameters view{};
  ProjectionParameters projection{};
  std::uint32_t width = 0;
  std::uint32_t height = 0;
};

inline CameraView makeCameraView(
  const Scene & scene, const TrainingCamera & camera, std::uint32_t width,
  bool zero_to_one_depth)
{
  CameraView out;
  out.width = width;
  out.height = std::uint32_t(std::lround(double(width) * camera.height / camera.width));
  const double near_clip = 0.02;
  const double far_clip = 1000.0;
  const double fovy = 2.0 * std::atan(camera.height / (2.0 * camera.fy));
  const double aspect = double(out.width) / double(out.height);

  // Ogre's camera looks down -z with y up; the 3DGS camera looks down +z with
  // y down, so its down and forward axes are negated.
  double view[4][4] = {};
  const double * c = camera.position;
  for (int column = 0; column < 3; ++column) {
    view[0][column] = camera.rotation[column][0];
    view[1][column] = -camera.rotation[column][1];
    view[2][column] = -camera.rotation[column][2];
  }
  for (int r = 0; r < 3; ++r) {
    view[r][3] = -(view[r][0] * c[0] + view[r][1] * c[1] + view[r][2] * c[2]);
  }
  view[3][3] = 1.0;

  const double focal = 1.0 / std::tan(fovy * 0.5);
  const auto projectionMatrix = [&](bool zero_to_one) {
      std::array<std::array<double, 4>, 4> p{};
      p[0][0] = focal / aspect;
      p[1][1] = focal;
      if (zero_to_one) {
        p[2][2] = far_clip / (near_clip - far_clip);
        p[2][3] = far_clip * near_clip / (near_clip - far_clip);
      } else {
        p[2][2] = (far_clip + near_clip) / (near_clip - far_clip);
        p[2][3] = 2.0 * far_clip * near_clip / (near_clip - far_clip);
      }
      p[3][2] = -1.0;
      return p;
    };
  const auto multiply = [&view](const std::array<std::array<double, 4>, 4> & p) {
      std::array<std::array<double, 4>, 4> m{};
      for (int r = 0; r < 4; ++r) {
        for (int k = 0; k < 4; ++k) {
          for (int j = 0; j < 4; ++j) {
            m[r][k] += p[r][j] * view[j][k];
          }
        }
      }
      return m;
    };

  ViewParameters & v = out.view;
  for (int r = 0; r < 4; ++r) {
    v.world_rows[r * 5] = 1.0f;
    for (int k = 0; k < 4; ++k) {
      v.view_rows[r * 4 + k] = float(view[r][k]);
    }
  }
  // Ogre's frustum planes, near, far, left, right, top, bottom, facing inward,
  // from the rows of the OpenGL-style view-projection.
  const auto m = multiply(projectionMatrix(false));
  const int plane_rows[6][2] = {{2, 1}, {2, -1}, {0, 1}, {0, -1}, {1, -1}, {1, 1}};
  for (int plane = 0; plane < 6; ++plane) {
    double n[4];
    for (int k = 0; k < 4; ++k) {
      n[k] = m[3][k] + plane_rows[plane][1] * m[plane_rows[plane][0]][k];
    }
    const double length = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int k = 0; k < 4; ++k) {
      v.frustum_planes[plane * 4 + k] = float(n[k] / length);
    }
  }
  for (int k = 0; k < 3; ++k) {
    v.local_camera[k] = float(c[k]);
    v.local_sort_direction[k] = float(camera.rotation[k][2] * -1.0);
  }
  v.world_scale = 1.0f;
  v.focal_y = float(double(out.height) * 0.5 / std::tan(fovy * 0.5));
  v.near_clip = float(near_clip);
  v.far_clip = float(far_clip);
  v.ortho_scale = 0.0f;
  v.sigma_radius = 3.0f;
  v.min_screen_radius = 0.0f;
  v.eps2d = 0.3f;
  v.count = scene.count;
  v.viewport_height = out.height;
  v.sh_coefficients = scene.coefficients;
  v.flags = kViewCulling | kViewPerspective;

  ProjectionParameters & q = out.projection;
  const auto clip = multiply(projectionMatrix(zero_to_one_depth));
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 4; ++k) {
      q.worldview_rows[r * 4 + k] = float(view[r][k]);
      q.worldviewproj_rows[r * 4 + k] = float(clip[r][k]);
    }
  }
  q.viewport_size[0] = float(out.width);
  q.viewport_size[1] = float(out.height);
  q.viewport_size[2] = 1.0f / float(out.width);
  q.viewport_size[3] = 1.0f / float(out.height);
  q.fovy = float(fovy);
  q.eps2d = 0.3f;
  q.antialiased = 0.0f;
  q.sigma_radius = 3.0f;
  return out;
}

// Mean and largest per-channel difference of two RGBA8 images, in 8-bit steps,
// the PSNR of their colour, and the share of pixels more than two steps apart.
struct ImageDifference
{
  double mean = 0.0;
  int largest = 0;
  double psnr = 0.0;
  double beyond_two = 0.0;
};

inline ImageDifference compareImages(
  const std::vector<std::uint8_t> & a, const std::vector<std::uint8_t> & b)
{
  ImageDifference d;
  double sum = 0.0, squares = 0.0;
  std::size_t far = 0;
  const std::size_t pixels = a.size() / 4;
  for (std::size_t p = 0; p < pixels; ++p) {
    int worst = 0;
    for (int c = 0; c < 4; ++c) {
      const int e = std::abs(int(a[p * 4 + c]) - int(b[p * 4 + c]));
      worst = std::max(worst, e);
      sum += e;
      if (c < 3) {
        squares += double(e) * e;
      }
    }
    d.largest = std::max(d.largest, worst);
    far += worst > 2;
  }
  d.mean = sum / double(a.size());
  const double mse = squares / double(pixels * 3);
  d.psnr = mse > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0;
  d.beyond_two = double(far) / double(pixels);
  return d;
}

// Half floats to 8-bit unorm, rounded and clamped as a render target would.
inline std::vector<std::uint8_t> halfToUnorm8(const std::vector<std::uint16_t> & halves)
{
  std::vector<std::uint8_t> out(halves.size());
  for (std::size_t i = 0; i < halves.size(); ++i) {
    const std::uint32_t h = halves[i];
    const std::uint32_t exponent = (h >> 10) & 0x1fu;
    const std::uint32_t mantissa = h & 0x3ffu;
    float value;
    if (exponent == 0) {
      value = std::ldexp(float(mantissa), -24);
    } else if (exponent == 31) {
      value = mantissa ? 0.0f : 65504.0f;
    } else {
      value = std::ldexp(float(mantissa | 0x400u), int(exponent) - 25);
    }
    if (h & 0x8000u) {
      value = -value;
    }
    out[i] = std::uint8_t(std::lround(std::min(std::max(value, 0.0f), 1.0f) * 255.0f));
  }
  return out;
}

// Writes premultiplied RGBA8 composited over black as a binary PPM, top row
// first; `bottom_up` for an image stored the OpenGL way.
inline void writePpm(
  const std::string & path, const std::vector<std::uint8_t> & rgba, std::uint32_t width,
  std::uint32_t height, bool bottom_up)
{
  std::FILE * file = std::fopen(path.c_str(), "wb");
  if (!file) {
    return;
  }
  std::fprintf(file, "P6\n%u %u\n255\n", width, height);
  std::vector<std::uint8_t> row(std::size_t(width) * 3);
  for (std::uint32_t y = 0; y < height; ++y) {
    const std::uint32_t source = bottom_up ? height - 1 - y : y;
    for (std::uint32_t x = 0; x < width; ++x) {
      for (int c = 0; c < 3; ++c) {
        row[x * 3 + c] = rgba[(std::size_t(source) * width + x) * 4 + c];
      }
    }
    std::fwrite(row.data(), 1, row.size(), file);
  }
  std::fclose(file);
}

}  // namespace verification
}  // namespace gaussian_splatting_rviz_plugins
