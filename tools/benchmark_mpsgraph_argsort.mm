// Benchmark Apple's GPU arg-sort without involving RViz or Ogre.
//
// Build:
//   clang++ -std=c++17 -O2 -fobjc-arc \
//     -framework Foundation -framework Metal \
//     -framework MetalPerformanceShadersGraph \
//     tools/benchmark_mpsgraph_argsort.mm -o /tmp/benchmark_mpsgraph_argsort
//
// Run (optional arguments are element counts):
//   /tmp/benchmark_mpsgraph_argsort 500000 1000000 1650000 3000000 5800000

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

struct Timings
{
  double compile_ms{};
  std::vector<double> run_ms;
  uint64_t allocated_before{};
  uint64_t allocated_after{};
};

uint32_t nextRandom(uint32_t & state)
{
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

double milliseconds(Clock::time_point begin, Clock::time_point end)
{
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

double percentile(std::vector<double> values, double fraction)
{
  std::sort(values.begin(), values.end());
  const auto position = static_cast<size_t>(
    std::round(fraction * static_cast<double>(values.size() - 1)));
  return values[position];
}

bool execute(
  MPSGraphExecutable * executable,
  id<MTLCommandQueue> queue,
  MPSGraphTensorData * input,
  MPSGraphTensorData * output,
  double & elapsed_ms)
{
  dispatch_semaphore_t completed = dispatch_semaphore_create(0);
  __block NSError * execution_error = nil;
  MPSGraphExecutableExecutionDescriptor * descriptor =
    [MPSGraphExecutableExecutionDescriptor new];
  descriptor.completionHandler = ^(
    NSArray<MPSGraphTensorData *> * results, NSError * error)
    {
      (void)results;
      execution_error = error;
      dispatch_semaphore_signal(completed);
    };

  const auto begin = Clock::now();
  [executable runAsyncWithMTLCommandQueue:queue
                              inputsArray:@[input]
                             resultsArray:@[output]
                      executionDescriptor:descriptor];
  dispatch_semaphore_wait(completed, DISPATCH_TIME_FOREVER);
  elapsed_ms = milliseconds(begin, Clock::now());

  if (execution_error != nil) {
    std::fprintf(stderr, "MPSGraph execution failed: %s\n",
      execution_error.localizedDescription.UTF8String);
    return false;
  }
  return true;
}

bool benchmarkCount(
  id<MTLDevice> device, id<MTLCommandQueue> queue, size_t count,
  int warmup_iterations, int measured_iterations, Timings & timings)
{
  @autoreleasepool {
    MPSShape * shape = @[@(count)];
    MPSGraph * graph = [MPSGraph new];
    MPSGraphTensor * keys = [graph placeholderWithShape:shape
                                               dataType:MPSDataTypeFloat32
                                                   name:@"depth_keys"];
    MPSGraphTensor * indices = [graph argSortWithTensor:keys
                                                   axis:0
                                             descending:YES
                                                   name:@"sorted_indices"];

    MPSGraphShapedType * input_type =
      [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
    MPSGraphDevice * graph_device = [MPSGraphDevice deviceWithMTLDevice:device];

    const auto compile_begin = Clock::now();
    MPSGraphExecutable * executable = [graph compileWithDevice:graph_device
                                                         feeds:@{keys: input_type}
                                                 targetTensors:@[indices]
                                              targetOperations:nil
                                         compilationDescriptor:nil];
    timings.compile_ms = milliseconds(compile_begin, Clock::now());
    if (executable == nil) {
      std::fprintf(stderr, "MPSGraph failed to compile for %zu elements\n", count);
      return false;
    }

    const NSUInteger byte_count = count * sizeof(float);
    id<MTLBuffer> input_buffer = [device newBufferWithLength:byte_count
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [device newBufferWithLength:count * sizeof(int32_t)
                                                      options:MTLResourceStorageModeShared];
    if (input_buffer == nil || output_buffer == nil) {
      std::fprintf(stderr, "Metal buffer allocation failed for %zu elements\n", count);
      return false;
    }

    auto * input_values = static_cast<float *>(input_buffer.contents);
    uint32_t random_state = 0x9e3779b9u ^ static_cast<uint32_t>(count);
    for (size_t i = 0; i < count; ++i) {
      // Positive finite values approximate transformed view-space depth keys.
      input_values[i] = static_cast<float>(nextRandom(random_state) & 0x00ffffffu);
    }

    MPSGraphTensorData * input_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                             shape:shape
                                          dataType:MPSDataTypeFloat32];
    MPSGraphTensorData * output_data =
      [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                             shape:shape
                                          dataType:MPSDataTypeInt32];

    timings.allocated_before = device.currentAllocatedSize;
    for (int i = 0; i < warmup_iterations; ++i) {
      double ignored = 0.0;
      if (!execute(executable, queue, input_data, output_data, ignored)) {
        return false;
      }
    }

    timings.run_ms.reserve(measured_iterations);
    for (int i = 0; i < measured_iterations; ++i) {
      double elapsed = 0.0;
      if (!execute(executable, queue, input_data, output_data, elapsed)) {
        return false;
      }
      timings.run_ms.push_back(elapsed);
    }
    timings.allocated_after = device.currentAllocatedSize;

    // Validate the complete index result once, outside the timed interval. This
    // also catches unsupported/no-op execution without adding a readback to runs.
    const auto * sorted_indices = static_cast<const int32_t *>(output_buffer.contents);
    for (size_t i = 0; i < count; ++i) {
      const int32_t index = sorted_indices[i];
      if (index < 0 || static_cast<size_t>(index) >= count) {
        std::fprintf(stderr, "Invalid sorted index %d at %zu\n", index, i);
        return false;
      }
      if (i > 0 && input_values[sorted_indices[i - 1]] < input_values[index]) {
        std::fprintf(stderr, "Result is not descending at %zu\n", i);
        return false;
      }
    }
  }
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  @autoreleasepool {
    if (@available(macOS 13.0, *)) {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) {
        std::fprintf(stderr, "No Metal device is available\n");
        return EXIT_FAILURE;
      }
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) {
        std::fprintf(stderr, "Could not create a Metal command queue\n");
        return EXIT_FAILURE;
      }

      std::vector<size_t> counts;
      for (int i = 1; i < argc; ++i) {
        char * end = nullptr;
        const unsigned long long value = std::strtoull(argv[i], &end, 10);
        if (end == argv[i] || *end != '\0' || value == 0) {
          std::fprintf(stderr, "Invalid element count: %s\n", argv[i]);
          return EXIT_FAILURE;
        }
        counts.push_back(static_cast<size_t>(value));
      }
      if (counts.empty()) {
        counts = {500000, 1000000, 1650000, 3000000, 5800000};
      }

      constexpr int kWarmups = 2;
      constexpr int kIterations = 7;
      std::printf("Device: %s\n", device.name.UTF8String);
      std::printf("MPSGraph argSort, descending float32 -> int32 indices\n");
      std::printf("Each steady run waits for GPU completion but does not copy results to CPU.\n");
      std::printf("elements,compile_ms,min_ms,median_ms,p95_ms,mean_ms,allocation_delta_MiB\n");

      for (const size_t count : counts) {
        Timings timings;
        if (!benchmarkCount(device, queue, count, kWarmups, kIterations, timings)) {
          return EXIT_FAILURE;
        }
        const double mean = std::accumulate(
          timings.run_ms.begin(), timings.run_ms.end(), 0.0) /
          static_cast<double>(timings.run_ms.size());
        const auto minimum = *std::min_element(timings.run_ms.begin(), timings.run_ms.end());
        const double allocation_delta_mib =
          (static_cast<double>(timings.allocated_after) -
          static_cast<double>(timings.allocated_before)) / (1024.0 * 1024.0);
        std::printf("%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f\n",
          count, timings.compile_ms, minimum,
          percentile(timings.run_ms, 0.50), percentile(timings.run_ms, 0.95),
          mean, allocation_delta_mib);
        std::fflush(stdout);
      }
      return EXIT_SUCCESS;
    }

    std::fprintf(stderr, "MPSGraph argSort requires macOS 13 or newer\n");
    return EXIT_FAILURE;
  }
}
