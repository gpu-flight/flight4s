#include <cstdio>
#include <cstring>
#include <vector>

#include "flight4s/cuda/cuda_driver.hpp"

namespace {

constexpr int skip_exit_code = 77;
constexpr int element_count = 256;
constexpr std::uint64_t size_bytes = element_count * sizeof(float);

}  // namespace

int main() {
  const flight4s::cuda::CudaDriver driver;
  const auto context_result = driver.retain_primary_context(0);
  if (!context_result.status.succeeded()) {
    std::printf(
        "skipping: no CUDA device (%s)\n",
        context_result.status.result_name.c_str());
    return skip_exit_code;
  }
  const auto context = context_result.context;
  int failures = 0;

  const auto stream_result = driver.create_stream(context, 1);
  if (!stream_result.status.succeeded()) {
    std::printf("stream creation failed\n");
    return 1;
  }
  const auto stream = stream_result.stream;

  const auto source_alloc = driver.allocate_pinned_memory(context, size_bytes);
  const auto destination_alloc = driver.allocate_pinned_memory(context, size_bytes);
  const auto device_alloc = driver.allocate_device_memory(context, size_bytes);
  if (!source_alloc.status.succeeded() || !destination_alloc.status.succeeded() ||
      !device_alloc.status.succeeded()) {
    std::printf("allocation failed\n");
    return 1;
  }
  auto* source = static_cast<float*>(source_alloc.address);
  auto* destination = static_cast<float*>(destination_alloc.address);
  for (int index = 0; index < element_count; ++index) {
    source[index] = static_cast<float>(index % 19 - 9);
    destination[index] = -999.0f;
  }

  // Capture a round trip through the device; nothing executes during capture.
  const auto begin = driver.begin_stream_capture(context, stream);
  if (!begin.succeeded()) {
    std::printf("capture begin failed: %s\n", begin.result_name.c_str());
    return 1;
  }
  const auto upload = driver.copy_host_to_device_async(
      context, device_alloc.address, 0, source, size_bytes, stream);
  const auto download = driver.copy_device_to_host_async(
      context, destination, device_alloc.address, 0, size_bytes, stream);
  const auto end = driver.end_stream_capture(context, stream);
  if (!upload.succeeded() || !download.succeeded() || !end.status.succeeded()) {
    std::printf("capture failed\n");
    return 1;
  }
  if (destination[0] != -999.0f) {
    std::printf("captured work executed during capture\n");
    ++failures;
  }
  const auto instantiated = driver.instantiate_graph(context, end.graph);
  if (!instantiated.status.succeeded()) {
    std::printf("instantiate failed: %s\n", instantiated.status.result_name.c_str());
    return 1;
  }

  for (int replay = 0; replay < 2; ++replay) {
    for (int index = 0; index < element_count; ++index) {
      source[index] = static_cast<float>(index % 19 - 9) + replay * 100.0f;
    }
    const auto launch = driver.launch_graph(context, instantiated.exec, stream);
    const auto synchronize = driver.synchronize_stream(context, stream);
    if (!launch.succeeded() || !synchronize.succeeded()) {
      std::printf("replay %d failed\n", replay);
      return 1;
    }
    if (std::memcmp(destination, source, size_bytes) != 0) {
      std::printf("replay %d produced wrong contents\n", replay);
      ++failures;
    }
  }

  const auto exec_destroy = driver.destroy_graph_exec(context, instantiated.exec);
  const auto graph_destroy = driver.destroy_graph(context, end.graph);
  const auto device_free = driver.free_device_memory(context, device_alloc.address);
  const auto source_free = driver.free_pinned_memory(context, source_alloc.address);
  const auto destination_free = driver.free_pinned_memory(context, destination_alloc.address);
  const auto stream_destroy = driver.destroy_stream(context, stream);
  const auto release = driver.release_primary_context(0);
  if (!exec_destroy.succeeded() || !graph_destroy.succeeded() ||
      !device_free.succeeded() || !source_free.succeeded() ||
      !destination_free.succeeded() || !stream_destroy.succeeded() ||
      !release.succeeded()) {
    std::printf("cleanup failed\n");
    ++failures;
  }

  if (failures == 0) {
    std::printf("cuda graph integration: all checks passed\n");
    return 0;
  }
  return 1;
}
