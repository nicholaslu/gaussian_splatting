#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <OgreHardwareVertexBuffer.h>
#include <OgreMaterial.h>
#include <OgrePrerequisites.h>
#include <OgreRenderTargetListener.h>
#include <OgreTexture.h>
#include <OgreVector.h>

#include "gaussian_splatting_msgs/msg/gaussian_splats.hpp"
#include "rviz_common/message_filter_display.hpp"
#include "rviz_common/properties/bool_property.hpp"
#include "rviz_common/properties/float_property.hpp"

namespace Ogre
{
class Rectangle2D;
}

namespace gaussian_splatting_rviz_plugins
{

class DepthSchemeResolver;
class GaussianSplatRenderable;
class MetalViewPreparation;

class GaussianSplattingDisplay
  : public rviz_common::MessageFilterDisplay<gaussian_splatting_msgs::msg::GaussianSplats>,
  public Ogre::RenderTargetListener
{
public:
  GaussianSplattingDisplay();
  ~GaussianSplattingDisplay() override;

  void onInitialize() override;
  void reset() override;
  void update(float wall_dt, float ros_dt) override;

  // Splats are drawn only while the offscreen target is being filled, and the
  // compositing quad only while the main window is. Ogre renders textures
  // (priority 2) before windows (priority 4), so the target is always ready.
  void preRenderTargetUpdate(const Ogre::RenderTargetEvent & event) override;
  void postRenderTargetUpdate(const Ogre::RenderTargetEvent & event) override;
  void preViewportUpdate(const Ogre::RenderTargetViewportEvent & event) override;

private:
  using GaussianSplats = gaussian_splatting_msgs::msg::GaussianSplats;

  void processMessage(GaussianSplats::ConstSharedPtr msg) override;
  void registerOgreResources();
  void clearMesh();

  // Returns false and sets the display status when the message violates the
  // array size invariants documented in GaussianSplats.msg.
  bool validate(const GaussianSplats & msg, std::size_t & count);

  // Allocates the mesh and its buffers for the given primitive count. Only
  // called when the count changes; a stream of equally sized messages reuses
  // the existing buffers via uploadSplats().
  void allocateMesh(std::size_t count);
  void uploadSplats(const GaussianSplats & msg, std::size_t count);

  // The per-splat record lives in a texture the vertex shader reads by index,
  // so a depth sort only has to rewrite 4 bytes per splat instead of moving
  // the whole record. Sized for the current splat count and filled once per
  // message; the draw order never touches it.
  void createSplatDataTexture(std::size_t count);
  void writeSplatDataTexture();
  void applyShaderParams();
  void writeSortedInstances();
  void sortIndexBuffer();

  // Offscreen rendering at a fraction of the viewport resolution. Splat cost is
  // dominated by fill rate, so this trades splat sharpness for roughly the
  // square of the scale factor in fragments.
  Ogre::Viewport * mainViewport() const;
  void updateRenderTarget();
  void destroyRenderTarget();
  void setSplatsVisible(bool visible);

  rviz_common::properties::FloatProperty * sigma_radius_property_ = nullptr;
  rviz_common::properties::BoolProperty * culling_property_ = nullptr;
  rviz_common::properties::FloatProperty * min_screen_radius_property_ = nullptr;
  rviz_common::properties::BoolProperty * offscreen_property_ = nullptr;
  rviz_common::properties::FloatProperty * render_scale_property_ = nullptr;

  Ogre::SceneNode * splat_node_ = nullptr;
  std::unique_ptr<GaussianSplatRenderable> renderable_;
  std::unique_ptr<MetalViewPreparation> metal_view_preparation_;
  Ogre::MaterialPtr material_;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;

  Ogre::TexturePtr splat_texture_;
  Ogre::TexturePtr splat_data_texture_;
  Ogre::Viewport * depth_viewport_ = nullptr;
  Ogre::Viewport * rtt_viewport_ = nullptr;
  Ogre::RenderTarget * main_target_ = nullptr;
  Ogre::Rectangle2D * composite_rect_ = nullptr;
  Ogre::SceneNode * composite_node_ = nullptr;
  Ogre::MaterialPtr composite_material_;
  std::unique_ptr<DepthSchemeResolver> depth_scheme_resolver_;
  unsigned int rtt_width_ = 0;
  unsigned int rtt_height_ = 0;
  double last_upload_ms_ = 0.0;
  double last_sort_ms_ = 0.0;
  double last_setup_ms_ = 0.0;
  double last_cull_sh_ms_ = 0.0;
  double last_merge_ms_ = 0.0;
  double last_radix_ms_ = 0.0;
  double last_gather_ms_ = 0.0;
  double last_instance_upload_ms_ = 0.0;

  std::size_t splat_count_ = 0;
  std::size_t visible_splat_count_ = 0;

  // One splat as the shader reads it, laid out as whole texels so a row of the
  // data texture is a whole number of splats. Colour is absent because it is
  // view dependent; it arrives per instance instead.
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

  std::vector<Ogre::Vector3> positions_;
  std::vector<SplatRecord> records_;
  std::vector<DrawInstance> draw_instances_;

  // Per-worker output for the cull pass, kept between frames so that a frame
  // does not allocate. Each worker fills its own vector and the results are
  // concatenated in worker order, which keeps the draw order deterministic.
  std::vector<std::vector<std::uint64_t>> cull_partitions_;

  // Spherical harmonics, needed until the camera stops moving rather than only
  // during processMessage, so the message is held and pointed into instead of
  // copied out of: at degree 3 the coefficients are 180 bytes a splat, and
  // copying them measured 6 ms against a memory bandwidth that leaves nothing
  // for threads to win. sh_coefficients_ is K-1, the count beyond the constant
  // term, so it is 0, 3, 8 or 15.
  GaussianSplats::ConstSharedPtr message_;
  const float * sh_dc_ = nullptr;
  const float * sh_rest_ = nullptr;
  std::size_t sh_coefficients_ = 0;

  // Three floats per splat index, filled for the splats that survive culling.
  std::vector<float> colours_;

  // Depth-sorted draw order, one word per surviving splat: the view depth as an
  // order-preserving key in the high 32 bits, the splat index in the low 32.
  // Packing them lets a single radix pass sort and permute at once, and keeps
  // the depth out of the comparator, where recomputing it was chasing a random
  // load per comparison.
  std::vector<std::uint64_t> order_;
  std::vector<std::uint64_t> order_scratch_;

  // Rasterisation convention carried by the most recent message.
  float eps2d_ = 0.3f;
  float antialiased_ = 0.0f;

  Ogre::Vector3 last_camera_position_ = Ogre::Vector3::ZERO;
  Ogre::Vector3 last_camera_direction_ = Ogre::Vector3::ZERO;
  unsigned int last_viewport_width_ = 0;
  unsigned int last_viewport_height_ = 0;
  float last_sigma_radius_ = -1.0f;
  float last_min_screen_radius_ = -1.0f;
  bool last_culling_enabled_ = false;
  bool using_gpu_preparation_ = false;
  bool sort_dirty_ = true;
  std::string mesh_name_;
  std::string material_name_;
  bool resources_registered_ = false;
};

}  // namespace gaussian_splatting_rviz_plugins
