#include "flight4s/cuda/cuda_driver.hpp"
#include "flight4s/cuda/current_context_scope.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flight4s::cuda {

CudaDriverStatus make_driver_status(
    CUresult result,
    std::string info_log,
    std::string error_log) {
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  return {
      result,
      name == nullptr ? "CUDA_ERROR_UNKNOWN" : name,
      description == nullptr ? "no CUDA error description" : description,
      std::move(info_log),
      std::move(error_log),
  };
}

namespace {

constexpr std::size_t jit_log_capacity = 64 * 1024;

void require_handle(const void* handle, const char* description) {
  if (handle == nullptr) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }
}

void require_device_address(
    CUdeviceptr address,
    const char* description) {
  if (address == 0) {
    throw std::invalid_argument(
        std::string(description) + " must not be null");
  }
}

CUdeviceptr checked_device_range(
    CUdeviceptr address,
    std::uint64_t offset_bytes,
    std::uint64_t size_bytes,
    const char* description) {
  const auto maximum = std::numeric_limits<CUdeviceptr>::max();
  if (offset_bytes > maximum - address) {
    throw std::overflow_error(
        std::string(description) + " exceeds the device address range");
  }
  const auto start = address + static_cast<CUdeviceptr>(offset_bytes);
  if (size_bytes - 1 > maximum - start) {
    throw std::overflow_error(
        std::string(description) + " extent exceeds the device address range");
  }
  return start;
}

void require_stream_flags(std::uint32_t flags) {
  if (flags != CU_STREAM_DEFAULT &&
      flags != CU_STREAM_NON_BLOCKING) {
    throw std::invalid_argument(
        "unsupported CUDA stream flags");
  }
}

void require_event_flags(std::uint32_t flags) {
  constexpr std::uint32_t supported_flags =
      CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING;
  if ((flags & ~supported_flags) != 0) {
    throw std::invalid_argument(
        "unsupported CUDA event flags");
  }
}

std::size_t checked_size(
    std::uint64_t size_bytes,
    const char* description) {
  if (size_bytes == 0) {
    throw std::invalid_argument(
        std::string(description) + " must be positive");
  }
  if (size_bytes > std::numeric_limits<std::size_t>::max()) {
    throw std::overflow_error(
        std::string(description) + " exceeds the native size limit");
  }
  return static_cast<std::size_t>(size_bytes);
}

void require_c_string(
    std::string_view value,
    const char* description) {
  if (value.empty()) {
    throw std::invalid_argument(
        std::string(description) + " must not be empty");
  }
  if (value.find('\0') != std::string_view::npos) {
    throw std::invalid_argument(
        std::string(description) + " must not contain a null byte");
  }
}

std::string log_text(const std::vector<char>& buffer) {
  const auto end = std::find(buffer.begin(), buffer.end(), '\0');
  return std::string(buffer.begin(), end);
}

CudaDriverStatus finish_context_operation(
    CUresult operation_result,
    CurrentContextScope& scope,
    std::string info_log = {},
    std::string error_log = {}) {
  const auto pop_result = scope.close();
  const auto final_result =
      operation_result == CUDA_SUCCESS ? pop_result : operation_result;
  return make_driver_status(
      final_result,
      std::move(info_log),
      std::move(error_log));
}

}  // namespace

CudaContextResult CudaDriver::retain_primary_context(
    std::int32_t device_ordinal) const {
  if (device_ordinal < 0) {
    throw std::invalid_argument(
        "CUDA device ordinal must not be negative");
  }

  auto result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  CUdevice device = 0;
  result = cuDeviceGet(&device, device_ordinal);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  int major = 0;
  result = cuDeviceGetAttribute(
      &major,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, 0, 0};
  }

  int minor = 0;
  result = cuDeviceGetAttribute(
      &minor,
      CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      device);
  if (result != CUDA_SUCCESS) {
    return {make_driver_status(result), nullptr, device_ordinal, major, 0};
  }

  CUcontext context = nullptr;
  result = cuDevicePrimaryCtxRetain(&context, device);
  return {
      make_driver_status(result),
      result == CUDA_SUCCESS ? context : nullptr,
      device_ordinal,
      major,
      minor,
  };
}

CudaDriverStatus CudaDriver::release_primary_context(
    std::int32_t device_ordinal) const {
  if (device_ordinal < 0) {
    throw std::invalid_argument(
        "CUDA device ordinal must not be negative");
  }

  CUdevice device = 0;
  const auto device_result = cuDeviceGet(&device, device_ordinal);
  if (device_result != CUDA_SUCCESS) {
    return make_driver_status(device_result);
  }
  return make_driver_status(cuDevicePrimaryCtxRelease(device));
}

CudaDriverStatus CudaDriver::synchronize_context(CUcontext context) const {
  require_handle(context, "CUDA context");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuCtxSynchronize(), current);
}

CudaModuleResult CudaDriver::load_ptx(
    CUcontext context,
    const std::vector<std::uint8_t>& ptx) const {
  require_handle(context, "CUDA context handle");
  if (ptx.empty()) {
    throw std::invalid_argument("PTX must not be empty");
  }
  if (std::find(ptx.begin(), ptx.end(), std::uint8_t{0}) != ptx.end()) {
    throw std::invalid_argument("PTX must not contain a null byte");
  }

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  std::vector<char> null_terminated_ptx(ptx.begin(), ptx.end());
  null_terminated_ptx.push_back('\0');
  std::vector<char> info_log(jit_log_capacity, '\0');
  std::vector<char> error_log(jit_log_capacity, '\0');
  std::array<CUjit_option, 4> options{
      CU_JIT_INFO_LOG_BUFFER,
      CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
      CU_JIT_ERROR_LOG_BUFFER,
      CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
  };
  std::array<void*, 4> option_values{
      info_log.data(),
      reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(info_log.size())),
      error_log.data(),
      reinterpret_cast<void*>(
          static_cast<std::uintptr_t>(error_log.size())),
  };

  CUmodule module = nullptr;
  const auto load_result = cuModuleLoadDataEx(
      &module,
      null_terminated_ptx.data(),
      static_cast<unsigned int>(options.size()),
      options.data(),
      option_values.data());
  auto status = finish_context_operation(
      load_result,
      current,
      log_text(info_log),
      log_text(error_log));
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? module : nullptr,
  };
}

CudaDriverStatus CudaDriver::unload_module(
    CUcontext context,
    CUmodule module) const {
  require_handle(context, "CUDA context handle");
  require_handle(module, "CUDA module handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuModuleUnload(module), current);
}

CudaFunctionResult CudaDriver::resolve_function(
    CUcontext context,
    CUmodule module,
    const std::string& name) const {
  require_handle(context, "CUDA context handle");
  require_handle(module, "CUDA module handle");
  require_c_string(name, "CUDA function name");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  CUfunction function = nullptr;
  const auto resolve_result =
      cuModuleGetFunction(&function, module, name.c_str());
  auto status =
      finish_context_operation(resolve_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? function : nullptr,
  };
}

CudaFunctionAttributesResult CudaDriver::function_attributes(
    CUcontext context,
    CUfunction function) const {
  require_handle(context, "CUDA context handle");
  require_handle(function, "CUDA function handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {
        make_driver_status(current.push_result()),
        {},
    };
  }

  int max_threads_per_block = 0;
  int static_shared_memory_bytes = 0;
  int constant_memory_bytes = 0;
  int local_memory_bytes = 0;
  int registers_per_thread = 0;

  auto query_result = cuFuncGetAttribute(
      &max_threads_per_block,
      CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
      function);
  if (query_result == CUDA_SUCCESS) {
    query_result = cuFuncGetAttribute(
        &static_shared_memory_bytes,
        CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
        function);
  }
  if (query_result == CUDA_SUCCESS) {
    query_result = cuFuncGetAttribute(
        &constant_memory_bytes,
        CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
        function);
  }
  if (query_result == CUDA_SUCCESS) {
    query_result = cuFuncGetAttribute(
        &local_memory_bytes,
        CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
        function);
  }
  if (query_result == CUDA_SUCCESS) {
    query_result = cuFuncGetAttribute(
        &registers_per_thread,
        CU_FUNC_ATTRIBUTE_NUM_REGS,
        function);
  }

  auto status = finish_context_operation(query_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded
          ? CudaFunctionAttributes{
                max_threads_per_block,
                static_shared_memory_bytes,
                constant_memory_bytes,
                local_memory_bytes,
                registers_per_thread,
            }
          : CudaFunctionAttributes{},
  };
}

CudaStreamResult CudaDriver::create_stream(
    CUcontext context,
    std::uint32_t flags) const {
  require_handle(context, "CUDA context handle");
  require_stream_flags(flags);

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  CUstream stream = nullptr;
  const auto create_result = cuStreamCreate(&stream, flags);
  auto status = finish_context_operation(create_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? stream : nullptr,
  };
}

CudaDriverStatus CudaDriver::destroy_stream(
    CUcontext context,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(stream, "CUDA stream handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuStreamDestroy(stream), current);
}

CudaDriverStatus CudaDriver::synchronize_stream(
    CUcontext context,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(stream, "CUDA stream handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuStreamSynchronize(stream),
      current);
}

CudaEventResult CudaDriver::create_event(
    CUcontext context,
    std::uint32_t flags) const {
  require_handle(context, "CUDA context handle");
  require_event_flags(flags);

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  CUevent event = nullptr;
  const auto create_result = cuEventCreate(&event, flags);
  auto status = finish_context_operation(create_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? event : nullptr,
  };
}

CudaDriverStatus CudaDriver::destroy_event(
    CUcontext context,
    CUevent event) const {
  require_handle(context, "CUDA context handle");
  require_handle(event, "CUDA event handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuEventDestroy(event), current);
}

CudaDriverStatus CudaDriver::record_event(
    CUcontext context,
    CUevent event,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(event, "CUDA event handle");
  require_handle(stream, "CUDA stream handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuEventRecord(event, stream),
      current);
}

CudaEventQueryResult CudaDriver::query_event(
    CUcontext context,
    CUevent event) const {
  require_handle(context, "CUDA context handle");
  require_handle(event, "CUDA event handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), false};
  }

  const auto query_result = cuEventQuery(event);
  if (query_result == CUDA_ERROR_NOT_READY) {
    return {
        finish_context_operation(CUDA_SUCCESS, current),
        false,
    };
  }
  auto status = finish_context_operation(query_result, current);
  const bool complete = status.succeeded();
  return {std::move(status), complete};
}

CudaDriverStatus CudaDriver::synchronize_event(
    CUcontext context,
    CUevent event) const {
  require_handle(context, "CUDA context handle");
  require_handle(event, "CUDA event handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuEventSynchronize(event),
      current);
}

CudaDriverStatus CudaDriver::wait_for_event(
    CUcontext context,
    CUstream stream,
    CUevent event) const {
  require_handle(context, "CUDA context handle");
  require_handle(stream, "CUDA stream handle");
  require_handle(event, "CUDA event handle");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuStreamWaitEvent(stream, event, 0),
      current);
}

CudaPinnedMemoryResult CudaDriver::allocate_pinned_memory(
    CUcontext context,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA pinned allocation size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), nullptr};
  }

  void* address = nullptr;
  const auto allocation_result = cuMemHostAlloc(
      &address,
      native_size,
      0);
  auto status = finish_context_operation(allocation_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? address : nullptr,
  };
}

CudaDriverStatus CudaDriver::free_pinned_memory(
    CUcontext context,
    void* address) const {
  require_handle(context, "CUDA context handle");
  require_handle(address, "CUDA pinned host address");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuMemFreeHost(address), current);
}

CudaDeviceMemoryResult CudaDriver::allocate_device_memory(
    CUcontext context,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA allocation size");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {make_driver_status(current.push_result()), 0};
  }

  CUdeviceptr address = 0;
  const auto allocation_result = cuMemAlloc(&address, native_size);
  auto status =
      finish_context_operation(allocation_result, current);
  const bool succeeded = status.succeeded();
  return {
      std::move(status),
      succeeded ? address : 0,
  };
}

CudaDriverStatus CudaDriver::free_device_memory(
    CUcontext context,
    CUdeviceptr address) const {
  require_handle(context, "CUDA context handle");
  require_device_address(address, "CUDA device address");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(cuMemFree(address), current);
}

CudaDriverStatus CudaDriver::copy_host_to_device(
    CUcontext context,
    CUdeviceptr destination,
    std::uint64_t destination_offset_bytes,
    const void* source,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  require_device_address(destination, "CUDA destination address");
  require_handle(source, "host source address");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");
  const auto destination_address = checked_device_range(
      destination,
      destination_offset_bytes,
      size_bytes,
      "CUDA destination offset");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  auto result = cuMemcpyHtoD(destination_address, source, native_size);
  if (result == CUDA_SUCCESS) {
    // Pageable HtoD may return after staging, before DMA reaches the device.
    result = cuStreamSynchronize(CU_STREAM_LEGACY);
  }
  return finish_context_operation(result, current);
}

CudaDriverStatus CudaDriver::copy_host_to_device_async(
    CUcontext context,
    CUdeviceptr destination,
    std::uint64_t destination_offset_bytes,
    const void* source,
    std::uint64_t size_bytes,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_device_address(destination, "CUDA destination address");
  require_handle(source, "host source address");
  require_handle(stream, "CUDA stream handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");
  const auto destination_address = checked_device_range(
      destination,
      destination_offset_bytes,
      size_bytes,
      "CUDA destination offset");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuMemcpyHtoDAsync(
          destination_address,
          source,
          native_size,
          stream),
      current);
}

CudaDriverStatus CudaDriver::copy_device_to_host(
    CUcontext context,
    void* destination,
    CUdeviceptr source,
    std::uint64_t source_offset_bytes,
    std::uint64_t size_bytes) const {
  require_handle(context, "CUDA context handle");
  require_handle(destination, "host destination address");
  require_device_address(source, "CUDA source address");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");
  const auto source_address = checked_device_range(
      source,
      source_offset_bytes,
      size_bytes,
      "CUDA source offset");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuMemcpyDtoH(destination, source_address, native_size),
      current);
}

CudaDriverStatus CudaDriver::copy_device_to_host_async(
    CUcontext context,
    void* destination,
    CUdeviceptr source,
    std::uint64_t source_offset_bytes,
    std::uint64_t size_bytes,
    CUstream stream) const {
  require_handle(context, "CUDA context handle");
  require_handle(destination, "host destination address");
  require_device_address(source, "CUDA source address");
  require_handle(stream, "CUDA stream handle");
  const auto native_size =
      checked_size(size_bytes, "CUDA copy size");
  const auto source_address = checked_device_range(
      source,
      source_offset_bytes,
      size_bytes,
      "CUDA source offset");

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return make_driver_status(current.push_result());
  }
  return finish_context_operation(
      cuMemcpyDtoHAsync(
          destination,
          source_address,
          native_size,
          stream),
      current);
}

}  // namespace flight4s::cuda
