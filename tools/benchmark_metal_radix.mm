// Sort-only benchmark of the GPU radix the display runs:
// MetalViewPreparationCore::encodeSort() on uniformly random keys, without the
// cull, compaction or gather stages. Run from the repository root.
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
//     -I gaussian_splatting_rviz_plugins/include -I gaussian_splatting_rviz_plugins/src \
//     tools/benchmark_metal_radix.mm \
//     gaussian_splatting_rviz_plugins/src/metal_view_preparation_core.mm \
//     -o /tmp/benchmark_metal_radix
//   /tmp/benchmark_metal_radix 500000 1000000 1650000 3000000 5800000
//
// GSPLAT_RADIX_BITS=16 sorts 16-bit keys in two passes instead of four. The
// radix is adapted from MetalSprocketsGaussianSplats; see
// gaussian_splatting_rviz_plugins/THIRD_PARTY_NOTICES.md.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "metal_view_preparation_core.h"

namespace
{
using Clock = std::chrono::steady_clock;
using gaussian_splatting_rviz_plugins::MetalViewPreparationCore;

constexpr const char * kShader =
  "gaussian_splatting_rviz_plugins/ogre_media/materials/programs/Metal/gsplat_prepare.metal";

struct Record {uint32_t key; uint32_t value;};
static_assert(sizeof(Record) == 8);

struct RunTime {double wall_ms; double gpu_ms;};

uint32_t nextRandom(uint32_t & state)
{
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

double percentile(std::vector<double> values, double fraction)
{
  std::sort(values.begin(), values.end());
  return values[static_cast<size_t>(
    std::round(fraction * static_cast<double>(values.size() - 1)))];
}
}  // namespace

int main(int argc, char ** argv)
{
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      std::fprintf(stderr, "No Metal device is available\n");
      return EXIT_FAILURE;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    MetalViewPreparationCore core(device, kShader);
    if (!core.ready()) {
      std::fprintf(stderr, "%s\n", core.error().c_str());
      return EXIT_FAILURE;
    }

    std::vector<size_t> counts;
    for (int i = 1; i < argc; ++i) {
      char * end = nullptr;
      const unsigned long long value = std::strtoull(argv[i], &end, 10);
      if (end == argv[i] || *end != '\0' || value == 0 || value > UINT32_MAX) {
        std::fprintf(stderr, "Invalid element count: %s\n", argv[i]);
        return EXIT_FAILURE;
      }
      counts.push_back(static_cast<size_t>(value));
    }
    if (counts.empty()) {counts = {500000, 1000000, 1650000, 3000000, 5800000};}
    constexpr int kWarmups = 2;
    constexpr int kIterations = 7;
    uint32_t key_bits = 32;
    if (const char * value = std::getenv("GSPLAT_RADIX_BITS")) {
      key_bits = static_cast<uint32_t>(std::strtoul(value, nullptr, 10));
    }
    if (key_bits != 16 && key_bits != 32) {
      std::fprintf(stderr, "GSPLAT_RADIX_BITS must be 16 or 32\n");
      return EXIT_FAILURE;
    }
    std::printf("Device: %s\n", device.name.UTF8String);
    std::printf("Stable %u-pass 8-bit Metal radix, %u-bit keys + uint32 indices\n",
      key_bits / 8, key_bits);
    std::printf("elements,gpu_min_ms,gpu_median_ms,gpu_p95_ms,wall_median_ms,scratch_MiB\n");

    for (const size_t count_size : counts) {
      @autoreleasepool {
        const uint32_t count = static_cast<uint32_t>(count_size);
        if (!core.configure(count, 6)) {
          std::fprintf(stderr, "%s\n", core.error().c_str());
          return EXIT_FAILURE;
        }
        std::vector<uint32_t> original(count_size);
        uint32_t state = 0x85ebca6bu ^ count;
        auto * records = static_cast<Record *>(core.keyBuffer().contents);
        for (size_t i = 0; i < count_size; ++i) {
          original[i] = nextRandom(state) & (key_bits == 16 ? 0xffffu : 0xffffffffu);
          records[i] = {original[i], static_cast<uint32_t>(i)};
        }
        core.primeSortInput(count);

        // Each pass sorts the key buffer in place (the pass count is even), so
        // runs after the first sort already-sorted input, as before this
        // benchmark moved onto the shipped kernels.
        const auto runSort = [&]() -> RunTime {
            const auto begin = Clock::now();
            id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
            command_buffer.label = @"3DGS radix sort benchmark";
            id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
            id<MTLBuffer> sorted = core.encodeSort(encoder, key_bits);
            [encoder endEncoding];
            [command_buffer commit];
            [command_buffer waitUntilCompleted];
            if (!sorted || command_buffer.error != nil) {
              std::fprintf(stderr, "Metal sort failed: %s\n", sorted ?
                command_buffer.error.localizedDescription.UTF8String : core.error().c_str());
              return {-1.0, -1.0};
            }
            const double gpu_ms = command_buffer.GPUEndTime > command_buffer.GPUStartTime ?
              (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0 : 0.0;
            return {std::chrono::duration<double, std::milli>(Clock::now() - begin).count(), gpu_ms};
          };

        for (int i = 0; i < kWarmups; ++i) {
          if (runSort().wall_ms < 0.0) {return EXIT_FAILURE;}
        }
        std::vector<double> gpu_times;
        std::vector<double> wall_times;
        for (int i = 0; i < kIterations; ++i) {
          const RunTime run = runSort();
          if (run.wall_ms < 0.0) {return EXIT_FAILURE;}
          gpu_times.push_back(run.gpu_ms);
          wall_times.push_back(run.wall_ms);
        }

        const auto * sorted = static_cast<const Record *>(core.keyBuffer().contents);
        for (size_t i = 0; i < count_size; ++i) {
          if (i > 0 && sorted[i - 1].key > sorted[i].key) {
            std::fprintf(stderr, "Keys are not sorted at %zu\n", i);
            return EXIT_FAILURE;
          }
          // Input indices ascend, so a stable sort leaves equal keys ascending too.
          if (i > 0 && sorted[i - 1].key == sorted[i].key && sorted[i - 1].value > sorted[i].value) {
            std::fprintf(stderr, "Sort is not stable at %zu\n", i);
            return EXIT_FAILURE;
          }
          if (sorted[i].value >= count || original[sorted[i].value] != sorted[i].key) {
            std::fprintf(stderr, "Key/index pair is corrupt at %zu\n", i);
            return EXIT_FAILURE;
          }
        }

        std::printf("%zu,%.3f,%.3f,%.3f,%.3f,%.1f\n", count_size,
          *std::min_element(gpu_times.begin(), gpu_times.end()),
          percentile(gpu_times, 0.50), percentile(gpu_times, 0.95),
          percentile(wall_times, 0.50), double(core.scratchBytes()) / (1024.0 * 1024.0));
        std::fflush(stdout);
      }
    }
    return EXIT_SUCCESS;
  }
}
