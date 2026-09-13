// Correctness/performance prototype for a stable 4-pass 8-bit Metal radix.
// Run from the repository root so the shader path below resolves.
//
//   clang++ -std=c++17 -O2 -fobjc-arc -framework Foundation -framework Metal \
//     tools/benchmark_metal_radix.mm -o /tmp/benchmark_metal_radix
//   /tmp/benchmark_metal_radix 500000 1000000 1650000 3000000 5800000

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace
{
using Clock = std::chrono::steady_clock;
constexpr NSUInteger kThreadsPerTile = 32;
constexpr NSUInteger kElementsPerTile = 1024;
constexpr NSUInteger kRadix = 256;

struct RadixParameters
{
  uint32_t count;
  uint32_t tile_count;
  uint32_t elements_per_tile;
  uint32_t shift;
};

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

bool reportError(NSError * error, const char * operation)
{
  if (error == nil) {return false;}
  std::fprintf(stderr, "%s failed: %s\n", operation, error.localizedDescription.UTF8String);
  return true;
}

RunTime runSort(
  id<MTLCommandQueue> queue,
  id<MTLComputePipelineState> histogram_pipeline,
  id<MTLComputePipelineState> offset_pipeline,
  id<MTLComputePipelineState> base_pipeline,
  id<MTLComputePipelineState> scatter_pipeline,
  id<MTLBuffer> records_a, id<MTLBuffer> records_b,
  id<MTLBuffer> histograms, id<MTLBuffer> offsets,
  id<MTLBuffer> totals, id<MTLBuffer> digit_bases,
  uint32_t count, uint32_t tile_count, uint32_t key_bits)
{
  const auto begin = Clock::now();
  id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
  command_buffer.label = @"3DGS radix sort benchmark";
  id<MTLBuffer> input = records_a;
  id<MTLBuffer> output = records_b;

  for (uint32_t shift = 0; shift < key_bits; shift += 8) {
    RadixParameters parameters{count, tile_count, kElementsPerTile, shift};

    id<MTLComputeCommandEncoder> histogram = [command_buffer computeCommandEncoder];
    [histogram setComputePipelineState:histogram_pipeline];
    [histogram setBuffer:input offset:0 atIndex:0];
    [histogram setBuffer:histograms offset:0 atIndex:1];
    [histogram setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [histogram dispatchThreadgroups:MTLSizeMake(tile_count, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    [histogram endEncoding];

    id<MTLComputeCommandEncoder> scan_offsets = [command_buffer computeCommandEncoder];
    [scan_offsets setComputePipelineState:offset_pipeline];
    [scan_offsets setBuffer:histograms offset:0 atIndex:0];
    [scan_offsets setBuffer:offsets offset:0 atIndex:1];
    [scan_offsets setBuffer:totals offset:0 atIndex:2];
    [scan_offsets setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [scan_offsets dispatchThreads:MTLSizeMake(kRadix, 1, 1)
               threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [scan_offsets endEncoding];

    id<MTLComputeCommandEncoder> scan_bases = [command_buffer computeCommandEncoder];
    [scan_bases setComputePipelineState:base_pipeline];
    [scan_bases setBuffer:totals offset:0 atIndex:0];
    [scan_bases setBuffer:digit_bases offset:0 atIndex:1];
    [scan_bases dispatchThreads:MTLSizeMake(1, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [scan_bases endEncoding];

    id<MTLComputeCommandEncoder> scatter = [command_buffer computeCommandEncoder];
    [scatter setComputePipelineState:scatter_pipeline];
    [scatter setBuffer:input offset:0 atIndex:0];
    [scatter setBuffer:output offset:0 atIndex:1];
    [scatter setBuffer:offsets offset:0 atIndex:2];
    [scatter setBuffer:digit_bases offset:0 atIndex:3];
    [scatter setBytes:&parameters length:sizeof(parameters) atIndex:4];
    [scatter dispatchThreadgroups:MTLSizeMake(tile_count, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(kThreadsPerTile, 1, 1)];
    [scatter endEncoding];
    std::swap(input, output);
  }

  [command_buffer commit];
  [command_buffer waitUntilCompleted];
  if (command_buffer.error != nil) {
    std::fprintf(stderr, "Metal command failed: %s\n",
      command_buffer.error.localizedDescription.UTF8String);
    return {-1.0, -1.0};
  }
  const double gpu_ms = command_buffer.GPUEndTime > command_buffer.GPUStartTime ?
    (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0 : 0.0;
  return {std::chrono::duration<double, std::milli>(Clock::now() - begin).count(), gpu_ms};
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
    NSError * error = nil;
    NSString * source = [NSString stringWithContentsOfFile:@"tools/metal_radix_sort.metal"
                                                  encoding:NSUTF8StringEncoding error:&error];
    if (source == nil || reportError(error, "Reading Metal source")) {return EXIT_FAILURE;}
    MTLCompileOptions * options = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {options.mathMode = MTLMathModeFast;}
    id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
    if (library == nil || reportError(error, "Compiling Metal source")) {return EXIT_FAILURE;}

    auto makePipeline = ^id<MTLComputePipelineState>(NSString * name) {
      NSError * pipeline_error = nil;
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:
        [library newFunctionWithName:name] error:&pipeline_error];
      reportError(pipeline_error, name.UTF8String);
      return pipeline;
    };
    id<MTLComputePipelineState> histogram = makePipeline(@"radix_histogram");
    id<MTLComputePipelineState> scan_offsets = makePipeline(@"radix_scan_offsets");
    id<MTLComputePipelineState> scan_bases = makePipeline(@"radix_scan_digit_bases");
    id<MTLComputePipelineState> scatter = makePipeline(@"radix_scatter");
    if (!histogram || !scan_offsets || !scan_bases || !scatter) {return EXIT_FAILURE;}

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
    std::printf("elements,gpu_min_ms,gpu_median_ms,gpu_p95_ms,wall_median_ms,buffer_MiB\n");

    for (const size_t count_size : counts) {
      @autoreleasepool {
        const uint32_t count = static_cast<uint32_t>(count_size);
        const uint32_t tile_count = (count + kElementsPerTile - 1) / kElementsPerTile;
        const NSUInteger record_bytes = count_size * sizeof(Record);
        const NSUInteger table_bytes =
          static_cast<NSUInteger>(tile_count) * kRadix * sizeof(uint32_t);
        const MTLResourceOptions shared = MTLResourceStorageModeShared;
        id<MTLBuffer> records_a = [device newBufferWithLength:record_bytes options:shared];
        id<MTLBuffer> records_b = [device newBufferWithLength:record_bytes options:shared];
        id<MTLBuffer> histograms = [device newBufferWithLength:table_bytes options:shared];
        id<MTLBuffer> offsets = [device newBufferWithLength:table_bytes options:shared];
        id<MTLBuffer> totals = [device newBufferWithLength:kRadix * sizeof(uint32_t) options:shared];
        id<MTLBuffer> digit_bases =
          [device newBufferWithLength:kRadix * sizeof(uint32_t) options:shared];
        if (!records_a || !records_b || !histograms || !offsets || !totals || !digit_bases) {
          std::fprintf(stderr, "Buffer allocation failed for %zu elements\n", count_size);
          return EXIT_FAILURE;
        }

        std::vector<uint32_t> original(count_size);
        uint32_t state = 0x85ebca6bu ^ count;
        auto * records = static_cast<Record *>(records_a.contents);
        for (size_t i = 0; i < count_size; ++i) {
          original[i] = nextRandom(state) & (key_bits == 16 ? 0xffffu : 0xffffffffu);
          records[i] = {original[i], static_cast<uint32_t>(i)};
        }

        for (int i = 0; i < kWarmups; ++i) {
          const RunTime run = runSort(queue, histogram, scan_offsets, scan_bases, scatter,
            records_a, records_b, histograms, offsets, totals, digit_bases,
            count, tile_count, key_bits);
          if (run.wall_ms < 0.0) {return EXIT_FAILURE;}
        }

        std::vector<double> gpu_times;
        std::vector<double> wall_times;
        for (int i = 0; i < kIterations; ++i) {
          const RunTime run = runSort(queue, histogram, scan_offsets, scan_bases, scatter,
            records_a, records_b, histograms, offsets, totals, digit_bases,
            count, tile_count, key_bits);
          if (run.wall_ms < 0.0) {return EXIT_FAILURE;}
          gpu_times.push_back(run.gpu_ms);
          wall_times.push_back(run.wall_ms);
        }

        const auto * sorted = static_cast<const Record *>(records_a.contents);
        for (size_t i = 0; i < count_size; ++i) {
          if (i > 0 && sorted[i - 1].key > sorted[i].key) {
            std::fprintf(stderr, "Keys are not sorted at %zu\n", i);
            return EXIT_FAILURE;
          }
          if (sorted[i].value >= count ||
            original[sorted[i].value] != sorted[i].key)
          {
            std::fprintf(stderr, "Key/index pair is corrupt at %zu\n", i);
            return EXIT_FAILURE;
          }
        }

        const double buffers_mib = static_cast<double>(
          record_bytes * 2 + table_bytes * 2 + kRadix * sizeof(uint32_t) * 2) /
          (1024.0 * 1024.0);
        std::printf("%zu,%.3f,%.3f,%.3f,%.3f,%.1f\n", count_size,
          *std::min_element(gpu_times.begin(), gpu_times.end()),
          percentile(gpu_times, 0.50), percentile(gpu_times, 0.95),
          percentile(wall_times, 0.50), buffers_mib);
        std::fflush(stdout);
      }
    }
    return EXIT_SUCCESS;
  }
}
