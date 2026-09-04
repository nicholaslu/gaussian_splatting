#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <OgreHardwareIndexBuffer.h>
#include <OgreHardwareVertexBuffer.h>
#include <OgreMaterial.h>
#include <OgreMesh.h>
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
  void writeDrawIndices();
  void sortIndexBuffer();

  // Offscreen rendering at a fraction of the viewport resolution. Splat cost is
  // dominated by fill rate, so this trades splat sharpness for roughly the
  // square of the scale factor in fragments.
  Ogre::Viewport * mainViewport() const;
  void updateRenderTarget();
  void destroyRenderTarget();
  void setSplatsVisible(bool visible);

  rviz_common::properties::FloatProperty * sigma_radius_property_ = nullptr;
  rviz_common::properties::BoolProperty * offscreen_property_ = nullptr;
  rviz_common::properties::FloatProperty * render_scale_property_ = nullptr;

  Ogre::SceneNode * splat_node_ = nullptr;
  Ogre::Entity * entity_ = nullptr;
  Ogre::MeshPtr mesh_;
  Ogre::MaterialPtr material_;
  Ogre::HardwareVertexBufferSharedPtr position_buffer_;
  Ogre::HardwareVertexBufferSharedPtr colour_buffer_;
  Ogre::HardwareVertexBufferSharedPtr scale_buffer_;
  Ogre::HardwareVertexBufferSharedPtr quat_buffer_;
  Ogre::HardwareIndexBufferSharedPtr index_buffer_;

  Ogre::TexturePtr splat_texture_;
  Ogre::Viewport * depth_viewport_ = nullptr;
  Ogre::Viewport * rtt_viewport_ = nullptr;
  Ogre::RenderTarget * main_target_ = nullptr;
  Ogre::Rectangle2D * composite_rect_ = nullptr;
  Ogre::SceneNode * composite_node_ = nullptr;
  Ogre::MaterialPtr composite_material_;
  unsigned int rtt_width_ = 0;
  unsigned int rtt_height_ = 0;

  std::size_t splat_count_ = 0;
  std::vector<Ogre::Vector3> positions_;
  std::vector<std::uint32_t> indices_;
  std::vector<std::uint32_t> draw_indices_;

  // Scratch buffers reused across messages to keep streaming allocation-free.
  std::vector<float> position_data_;
  std::vector<float> colour_data_;
  std::vector<float> scale_data_;
  std::vector<float> quat_data_;

  // Rasterisation convention carried by the most recent message.
  float eps2d_ = 0.3f;
  float antialiased_ = 0.0f;

  Ogre::Vector3 last_camera_position_ = Ogre::Vector3::ZERO;
  Ogre::Vector3 last_camera_direction_ = Ogre::Vector3::ZERO;
  std::string mesh_name_;
  std::string material_name_;
  bool resources_registered_ = false;
};

}  // namespace gaussian_splatting_rviz_plugins
