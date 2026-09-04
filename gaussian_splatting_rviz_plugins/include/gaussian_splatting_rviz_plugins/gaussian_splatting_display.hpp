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
  void applyShaderParams();
  void writeSortedInstances();
  void sortIndexBuffer();

  // Offscreen rendering at a fraction of the viewport resolution. Splat cost is
  // dominated by fill rate, so this trades splat sharpness for roughly the
  // square of the scale factor in fragments.
  Ogre::Viewport * mainViewport() const;
  void updateAutomaticRenderScale(float wall_dt);
  void updateRenderTarget();
  void destroyRenderTarget();
  void setSplatsVisible(bool visible);

  rviz_common::properties::FloatProperty * sigma_radius_property_ = nullptr;
  rviz_common::properties::BoolProperty * culling_property_ = nullptr;
  rviz_common::properties::FloatProperty * min_screen_radius_property_ = nullptr;
  rviz_common::properties::BoolProperty * offscreen_property_ = nullptr;
  rviz_common::properties::FloatProperty * render_scale_property_ = nullptr;
  rviz_common::properties::BoolProperty * automatic_render_scale_property_ = nullptr;
  rviz_common::properties::FloatProperty * minimum_render_scale_property_ = nullptr;
  rviz_common::properties::FloatProperty * target_frame_rate_property_ = nullptr;

  Ogre::SceneNode * splat_node_ = nullptr;
  std::unique_ptr<GaussianSplatRenderable> renderable_;
  Ogre::MaterialPtr material_;
  Ogre::HardwareVertexBufferSharedPtr instance_buffer_;

  Ogre::TexturePtr splat_texture_;
  Ogre::Viewport * depth_viewport_ = nullptr;
  Ogre::Viewport * rtt_viewport_ = nullptr;
  Ogre::RenderTarget * main_target_ = nullptr;
  Ogre::Rectangle2D * composite_rect_ = nullptr;
  Ogre::SceneNode * composite_node_ = nullptr;
  Ogre::MaterialPtr composite_material_;
  std::unique_ptr<DepthSchemeResolver> depth_scheme_resolver_;
  unsigned int rtt_width_ = 0;
  unsigned int rtt_height_ = 0;
  float effective_render_scale_ = 1.0f;
  float automatic_scale_elapsed_ = 0.0f;
  bool automatic_scale_was_enabled_ = false;
  double last_upload_ms_ = 0.0;
  double last_sort_ms_ = 0.0;

  std::size_t splat_count_ = 0;
  std::size_t visible_splat_count_ = 0;
  struct SplatInstance
  {
    float position[3];
    float colour[4];
    float scale[3];
    float quat[4];
  };
  std::vector<Ogre::Vector3> positions_;
  std::vector<std::uint32_t> indices_;
  std::vector<SplatInstance> instances_;
  std::vector<SplatInstance> sorted_instances_;

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
  bool sort_dirty_ = true;
  std::string mesh_name_;
  std::string material_name_;
  bool resources_registered_ = false;
};

}  // namespace gaussian_splatting_rviz_plugins
