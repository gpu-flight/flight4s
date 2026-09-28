#pragma once

#include <cuda.h>

#include <cstdint>
#include <string>
#include <vector>

namespace flight4s::cuda {

struct CudaDriverStatus {
  CUresult result_code;
  std::string result_name;
  std::string result_description;
  std::string info_log;
  std::string error_log;

  [[nodiscard]] bool succeeded() const noexcept {
    return result_code == CUDA_SUCCESS;
  }
};

[[nodiscard]] CudaDriverStatus make_driver_status(
    CUresult result,
    std::string info_log = {},
    std::string error_log = {});

struct CudaContextResult {
  CudaDriverStatus status;
  CUcontext context;
  std::int32_t device_ordinal;
  std::int32_t compute_capability_major;
  std::int32_t compute_capability_minor;
};

struct CudaModuleResult {
  CudaDriverStatus status;
  CUmodule module;
};

struct CudaFunctionResult {
  CudaDriverStatus status;
  CUfunction function;
};

struct CudaFunctionAttributes {
  std::int32_t max_threads_per_block;
  std::int32_t static_shared_memory_bytes;
  std::int32_t constant_memory_bytes;
  std::int32_t local_memory_bytes;
  std::int32_t registers_per_thread;
};

struct CudaFunctionAttributesResult {
  CudaDriverStatus status;
  CudaFunctionAttributes attributes;
};

struct CudaDeviceMemoryResult {
  CudaDriverStatus status;
  CUdeviceptr address;
};

struct CudaPinnedMemoryResult {
  CudaDriverStatus status;
  void* address;
};

struct CudaStreamResult {
  CudaDriverStatus status;
  CUstream stream;
};

struct CudaEventResult {
  CudaDriverStatus status;
  CUevent event;
};

struct CudaEventQueryResult {
  CudaDriverStatus status;
  bool complete;
};

class CudaDriver final {
 public:
  [[nodiscard]] CudaContextResult retain_primary_context(
      std::int32_t device_ordinal) const;

  [[nodiscard]] CudaDriverStatus release_primary_context(
      std::int32_t device_ordinal) const;

  [[nodiscard]] CudaDriverStatus synchronize_context(
      CUcontext context) const;

  [[nodiscard]] CudaModuleResult load_ptx(
      CUcontext context,
      const std::vector<std::uint8_t>& ptx) const;

  [[nodiscard]] CudaDriverStatus unload_module(
      CUcontext context,
      CUmodule module) const;

  [[nodiscard]] CudaFunctionResult resolve_function(
      CUcontext context,
      CUmodule module,
      const std::string& name) const;

  [[nodiscard]] CudaFunctionAttributesResult function_attributes(
      CUcontext context,
      CUfunction function) const;

  [[nodiscard]] CudaStreamResult create_stream(
      CUcontext context,
      std::uint32_t flags) const;

  [[nodiscard]] CudaDriverStatus destroy_stream(
      CUcontext context,
      CUstream stream) const;

  [[nodiscard]] CudaDriverStatus synchronize_stream(
      CUcontext context,
      CUstream stream) const;

  [[nodiscard]] CudaEventResult create_event(
      CUcontext context,
      std::uint32_t flags) const;

  [[nodiscard]] CudaDriverStatus destroy_event(
      CUcontext context,
      CUevent event) const;

  [[nodiscard]] CudaDriverStatus record_event(
      CUcontext context,
      CUevent event,
      CUstream stream) const;

  [[nodiscard]] CudaEventQueryResult query_event(
      CUcontext context,
      CUevent event) const;

  [[nodiscard]] CudaDriverStatus synchronize_event(
      CUcontext context,
      CUevent event) const;

  [[nodiscard]] CudaDriverStatus wait_for_event(
      CUcontext context,
      CUstream stream,
      CUevent event) const;

  [[nodiscard]] CudaPinnedMemoryResult allocate_pinned_memory(
      CUcontext context,
      std::uint64_t size_bytes) const;

  [[nodiscard]] CudaDriverStatus free_pinned_memory(
      CUcontext context,
      void* address) const;

  [[nodiscard]] CudaDeviceMemoryResult allocate_device_memory(
      CUcontext context,
      std::uint64_t size_bytes) const;

  [[nodiscard]] CudaDriverStatus free_device_memory(
      CUcontext context,
      CUdeviceptr address) const;

  // Success means the upload is complete, including for pageable host memory.
  [[nodiscard]] CudaDriverStatus copy_host_to_device(
      CUcontext context,
      CUdeviceptr destination,
      std::uint64_t destination_offset_bytes,
      const void* source,
      std::uint64_t size_bytes) const;

  [[nodiscard]] CudaDriverStatus copy_host_to_device_async(
      CUcontext context,
      CUdeviceptr destination,
      std::uint64_t destination_offset_bytes,
      const void* source,
      std::uint64_t size_bytes,
      CUstream stream) const;

  [[nodiscard]] CudaDriverStatus copy_device_to_host(
      CUcontext context,
      void* destination,
      CUdeviceptr source,
      std::uint64_t source_offset_bytes,
      std::uint64_t size_bytes) const;

  [[nodiscard]] CudaDriverStatus copy_device_to_host_async(
      CUcontext context,
      void* destination,
      CUdeviceptr source,
      std::uint64_t source_offset_bytes,
      std::uint64_t size_bytes,
      CUstream stream) const;
};

}  // namespace flight4s::cuda
