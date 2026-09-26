// Compares the two ways of drawing a prepared view on Metal: the render
// pipeline the display uses today, a blended quad per splat through
// gsplat_projected_vp and gsplat_fp from gsplat.metal, and the compute tile
// rasteriser, which composites front to back and stops each pixel once it is
// opaque. Both draw the same prepared instances of a trained scene from one of
// its training cameras; the bench reports how far apart the images are and the
// GPU time of each.
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
//     -I gaussian_splatting_rviz_plugins/include -I gaussian_splatting_rviz_plugins/src \
//     gaussian_splatting_rviz_plugins/test/bench_tile_raster.mm \
//     gaussian_splatting_rviz_plugins/src/metal_view_preparation_core.mm -o /tmp/bench_tile_raster
//   /tmp/bench_tile_raster gaussian_splatting_rviz_plugins/ogre_media/materials/programs/Metal \
//     dataset/pretrained_models/garden [--camera N]... [--width W]... [--out DIR]

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gaussian_splatting_rviz_plugins/splat_view.hpp"
#include "metal_view_preparation_core.h"
#include "preparation_test_scenes.hpp"
#include "real_scene.hpp"

namespace
{
using namespace gaussian_splatting_rviz_plugins::verification;
using gaussian_splatting_rviz_plugins::MetalViewPreparationCore;
using gaussian_splatting_rviz_plugins::ProjectedInstance;

double gpuMilliseconds(id<MTLCommandBuffer> command_buffer)
{
  return (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
}

// The display's material for GPU-prepared splats: the vertex program places
// each instance's quad, the fragment program shades it, and the pass blends
// premultiplied colour with "one one_minus_src_alpha" into a cleared target.
id<MTLRenderPipelineState> quadPipeline(
  id<MTLDevice> device, const std::string & directory, MTLPixelFormat format)
{
  NSError * error = nil;
  NSString * body = [NSString stringWithContentsOfFile:
    [NSString stringWithUTF8String:(directory + "/gsplat.metal").c_str()]
    encoding:NSUTF8StringEncoding error:&error];
  if (!body) {
    std::fprintf(stderr, "reading gsplat.metal: %s\n", error.localizedDescription.UTF8String);
    return nil;
  }
  // What Ogre's Metal render system puts in front of a program's source.
  NSString * source = [@"#include <metal_stdlib>\n#define CONST_SLOT_START 16\n"
    stringByAppendingString:body];
  id<MTLLibrary> library = [device newLibraryWithSource:source options:[MTLCompileOptions new]
                                                  error:&error];
  if (!library) {
    std::fprintf(stderr, "compiling gsplat.metal: %s\n", error.localizedDescription.UTF8String);
    return nil;
  }
  MTLVertexDescriptor * vertices = [MTLVertexDescriptor vertexDescriptor];
  vertices.attributes[10].format = MTLVertexFormatFloat2;
  vertices.attributes[10].bufferIndex = 0;
  vertices.layouts[0].stride = sizeof(float) * 2;
  const NSUInteger instance_attributes[3][2] = {{8, 0}, {9, 16}, {3, 32}};
  for (const auto & attribute : instance_attributes) {
    vertices.attributes[attribute[0]].format = MTLVertexFormatFloat4;
    vertices.attributes[attribute[0]].offset = attribute[1];
    vertices.attributes[attribute[0]].bufferIndex = 1;
  }
  vertices.layouts[1].stride = sizeof(ProjectedInstance);
  vertices.layouts[1].stepFunction = MTLVertexStepFunctionPerInstance;

  MTLRenderPipelineDescriptor * descriptor = [MTLRenderPipelineDescriptor new];
  descriptor.vertexFunction = [library newFunctionWithName:@"gsplat_projected_vp"];
  descriptor.fragmentFunction = [library newFunctionWithName:@"gsplat_fp"];
  descriptor.vertexDescriptor = vertices;
  MTLRenderPipelineColorAttachmentDescriptor * colour = descriptor.colorAttachments[0];
  colour.pixelFormat = format;
  colour.blendingEnabled = YES;
  colour.sourceRGBBlendFactor = MTLBlendFactorOne;
  colour.sourceAlphaBlendFactor = MTLBlendFactorOne;
  colour.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  colour.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
  id<MTLRenderPipelineState> pipeline =
    [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
  if (!pipeline) {
    std::fprintf(stderr, "quad pipeline: %s\n", error.localizedDescription.UTF8String);
  }
  return pipeline;
}

id<MTLTexture> target(
  id<MTLDevice> device, std::uint32_t width, std::uint32_t height,
  MTLPixelFormat format = MTLPixelFormatRGBA8Unorm)
{
  MTLTextureDescriptor * descriptor =
    [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                       width:width height:height mipmapped:NO];
  descriptor.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
    MTLTextureUsageShaderWrite;
  descriptor.storageMode = MTLStorageModeShared;
  return [device newTextureWithDescriptor:descriptor];
}

std::vector<std::uint16_t> readHalfTexture(id<MTLTexture> texture)
{
  std::vector<std::uint16_t> halves(texture.width * texture.height * 4);
  [texture getBytes:halves.data() bytesPerRow:texture.width * 8
         fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height) mipmapLevel:0];
  return halves;
}

std::vector<std::uint8_t> readTexture(id<MTLTexture> texture)
{
  std::vector<std::uint8_t> pixels(texture.width * texture.height * 4);
  [texture getBytes:pixels.data() bytesPerRow:texture.width * 4
         fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height) mipmapLevel:0];
  return pixels;
}

}  // namespace

int main(int argc, char ** argv)
{
  @autoreleasepool {
    if (argc < 3) {
      std::fprintf(stderr, "usage: %s METAL_PROGRAM_DIR MODEL_DIR [--camera N]... [--width W]... "
        "[--capacity PAIRS_PER_SPLAT] [--out DIR]\n", argv[0]);
      return 2;
    }
    const std::string programs = argv[1];
    const std::string model = argv[2];
    std::vector<int> cameras;
    std::vector<std::uint32_t> widths;
    std::vector<double> forwards;
    double capacity_per_splat = 8.0;
    std::string out;
    for (int i = 3; i + 1 < argc; i += 2) {
      if (std::strcmp(argv[i], "--camera") == 0) {
        cameras.push_back(std::atoi(argv[i + 1]));
      } else if (std::strcmp(argv[i], "--width") == 0) {
        widths.push_back(std::uint32_t(std::atoi(argv[i + 1])));
      } else if (std::strcmp(argv[i], "--forward") == 0) {
        forwards.push_back(std::atof(argv[i + 1]));
      } else if (std::strcmp(argv[i], "--capacity") == 0) {
        capacity_per_splat = std::atof(argv[i + 1]);
      } else if (std::strcmp(argv[i], "--out") == 0) {
        out = argv[i + 1];
      }
    }
    if (cameras.empty()) {
      cameras = {0};
    }
    if (widths.empty()) {
      widths = {1297};
    }
    if (forwards.empty()) {
      forwards = {0.0};
    }

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      std::printf("no Metal device; skipping\n");
      return kSkipped;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MetalViewPreparationCore core(device, programs + "/gsplat_prepare.metal");
    id<MTLRenderPipelineState> quads = quadPipeline(device, programs, MTLPixelFormatRGBA8Unorm);
    id<MTLRenderPipelineState> half_quads =
      quadPipeline(device, programs, MTLPixelFormatRGBA16Float);
    if (!core.ready() || !quads || !half_quads) {
      std::fprintf(stderr, "setup failed: %s\n", core.error().c_str());
      return 1;
    }

    Scene scene;
    std::string error;
    const auto load_start = std::chrono::steady_clock::now();
    if (!loadPly(model + "/point_cloud/iteration_30000/point_cloud.ply", scene, error)) {
      std::fprintf(stderr, "%s\n", error.c_str());
      return 1;
    }
    std::printf("%s: %u splats, degree %u SH, loaded in %.1f s\n", device.name.UTF8String,
      scene.count, scene.coefficients == 15 ? 3u : scene.coefficients == 8 ? 2u :
      scene.coefficients == 3 ? 1u : 0u,
      milliseconds(std::chrono::steady_clock::now() - load_start) / 1000.0);
    if (!core.configure(scene.count, kIndexCount) ||
      !core.uploadStaticData(scene.records.data(), scene.records.size(), scene.dc.data(),
      scene.dc.size(), scene.rest.data(), scene.rest.size(), nil))
    {
      std::fprintf(stderr, "upload failed: %s\n", core.error().c_str());
      return 1;
    }
    id<MTLBuffer> instances = [device
      newBufferWithLength:std::size_t(scene.count) * sizeof(ProjectedInstance)
                  options:MTLResourceStorageModePrivate];
    id<MTLBuffer> draw = [device newBufferWithLength:sizeof(DrawIndexedArguments)
                                             options:MTLResourceStorageModeShared];
    const float corners[8] = {-1, -1, 1, -1, 1, 1, -1, 1};
    const std::uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
    id<MTLBuffer> corner_buffer = [device newBufferWithBytes:corners length:sizeof(corners)
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> index_buffer = [device newBufferWithBytes:indices length:sizeof(indices)
                                                    options:MTLResourceStorageModeShared];

    const auto run = [&](auto && body) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        body(command_buffer);
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted) {
          std::fprintf(stderr, "command buffer failed: %s\n",
            command_buffer.error.localizedDescription.UTF8String);
          std::exit(1);
        }
        return gpuMilliseconds(command_buffer);
      };
    const auto compute = [&](auto && body) {
        return run([&](id<MTLCommandBuffer> command_buffer) {
                 id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
                 body(encoder);
                 [encoder endEncoding];
               });
      };
    constexpr int kRepeats = 9;

    for (int camera_index : cameras) {
     for (double forward : forwards) {
      TrainingCamera camera;
      if (!loadCamera(model + "/cameras.json", camera_index, camera, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
      }
      moveForward(camera, forward);
      for (std::uint32_t width : widths) {
        const CameraView view = makeCameraView(scene, camera, width, true);
        const auto capacity = std::uint32_t(std::min(
            double(scene.count) * capacity_per_splat, double(0xffffffffu / 2u)));
        if (!core.configureRaster(view.width, view.height, capacity, true, 0.0f)) {
          std::fprintf(stderr, "raster setup failed: %s\n", core.error().c_str());
          return 1;
        }
        id<MTLTexture> quad_target = target(device, view.width, view.height);
        id<MTLTexture> tile_target = target(device, view.width, view.height);
        id<MTLTexture> half_target =
          target(device, view.width, view.height, MTLPixelFormatRGBA16Float);

        std::vector<double> prepare, quad, tile, bin, sort, tiles;
        id<MTLTexture> quad_texture = quad_target;
        id<MTLRenderPipelineState> quad_pipeline = quads;
        const auto draw_quads = [&](id<MTLCommandBuffer> command_buffer) {
            MTLRenderPassDescriptor * pass = [MTLRenderPassDescriptor renderPassDescriptor];
            pass.colorAttachments[0].texture = quad_texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0);
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            id<MTLRenderCommandEncoder> encoder =
              [command_buffer renderCommandEncoderWithDescriptor:pass];
            [encoder setRenderPipelineState:quad_pipeline];
            [encoder setVertexBuffer:corner_buffer offset:0 atIndex:0];
            [encoder setVertexBuffer:instances offset:0 atIndex:1];
            [encoder drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexType:MTLIndexTypeUInt16
                               indexBuffer:index_buffer indexBufferOffset:0 indirectBuffer:draw
                      indirectBufferOffset:0];
            [encoder endEncoding];
          };
        for (int rep = 0; rep < kRepeats; ++rep) {
          prepare.push_back(compute([&](id<MTLComputeCommandEncoder> encoder) {
              core.encode(encoder, view.view, view.projection, instances, draw);
            }));
          quad.push_back(run(draw_quads));
          tile.push_back(compute([&](id<MTLComputeCommandEncoder> encoder) {
              core.encodeRaster(encoder, instances, tile_target);
            }));
          id<MTLBuffer> sorted = nil;
          bin.push_back(compute([&](id<MTLComputeCommandEncoder> encoder) {
              core.encodeRasterBin(encoder, instances);
            }));
          sort.push_back(compute([&](id<MTLComputeCommandEncoder> encoder) {
              sorted = core.encodeRasterSort(encoder);
            }));
          tiles.push_back(compute([&](id<MTLComputeCommandEncoder> encoder) {
              core.encodeRasterTiles(encoder, sorted, instances, tile_target);
            }));
        }

        quad_texture = half_target;
        quad_pipeline = half_quads;
        std::vector<double> half_quad;
        for (int rep = 0; rep < kRepeats; ++rep) {
          half_quad.push_back(run(draw_quads));
        }
        const std::vector<std::uint8_t> half_image = halfToUnorm8(readHalfTexture(half_target));
        const auto * state =
          static_cast<const MetalViewPreparationCore::State *>(core.stateBuffer().contents);
        const auto * raster =
          static_cast<const MetalViewPreparationCore::State *>(core.rasterStateBuffer().contents);
        const std::vector<std::uint8_t> quad_image = readTexture(quad_target);
        const std::vector<std::uint8_t> tile_image = readTexture(tile_target);
        const ImageDifference difference = compareImages(quad_image, tile_image);
        const ImageDifference quad_error = compareImages(half_image, quad_image);
        const ImageDifference tile_error = compareImages(half_image, tile_image);
        std::printf(
          "\ncamera %d, %.1f forward, at %ux%u: %u of %u visible, %u tile pairs (%.2f per visible splat)%s\n",
          camera_index, forward, view.width, view.height, state->visible, scene.count, raster->reserved,
          state->visible ? double(raster->reserved) / state->visible : 0.0,
          raster->reserved > capacity ? "  OVERFLOW: raise --capacity" : "");
        std::printf("  preparation (unchanged)          %7.2f ms\n", median(prepare));
        std::printf("  quads through the render pipeline %7.2f ms\n", median(quad));
        std::printf("  quads into a half float target    %7.2f ms\n", median(half_quad));
        std::printf("  tile rasteriser                   %7.2f ms  (bin %.2f, sort %.2f, "
          "tiles %.2f)\n", median(tile), median(bin), median(sort), median(tiles));
        std::printf("  images: mean |d| %.3f, largest %d, PSNR %.1f dB, %.3f%% of pixels > 2 "
          "steps apart\n", difference.mean, difference.largest, difference.psnr,
          100.0 * difference.beyond_two);
        std::printf("  against quads blended in half float: quads PSNR %.1f dB (%.3f%% > 2 steps), "
          "tiles PSNR %.1f dB (%.3f%% > 2 steps)\n", quad_error.psnr, 100.0 * quad_error.beyond_two,
          tile_error.psnr, 100.0 * tile_error.beyond_two);
        if (!out.empty()) {
          const std::string stem = out + "/metal_camera" + std::to_string(camera_index) + "_f" +
            std::to_string(int(forward * 10)) + "_" + std::to_string(view.width);
          writePpm(stem + "_quads.ppm", quad_image, view.width, view.height, false);
          writePpm(stem + "_tiles.ppm", tile_image, view.width, view.height, false);
        }
      }
     }
    }
    return 0;
  }
}
