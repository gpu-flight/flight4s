#include "flight4s/cuda/cuda_driver.hpp"
#include "flight4s/cuda/current_context_scope.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require_success(
    const flight4s::cuda::CudaDriverStatus& status,
    const char* operation) {
  if (status.succeeded()) {
    return;
  }
  throw std::runtime_error(
      std::string(operation) + " failed with " +
      status.result_name + ": " + status.result_description +
      "\n" + status.error_log);
}

std::vector<std::uint8_t> read_file(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("could not open PTX test file");
  }
  return {
      std::istreambuf_iterator<char>(input),
      std::istreambuf_iterator<char>(),
  };
}

}  // namespace

int main(int argument_count, char** arguments) {
  if (argument_count != 2) {
    std::cerr << "expected PTX path\n";
    return 1;
  }

  const flight4s::cuda::CudaDriver driver;
  CUcontext context = nullptr;
  CUmodule module = nullptr;
  CUstream stream = nullptr;
  CUevent event = nullptr;
  void* pinned_memory = nullptr;
  CUdeviceptr memory = 0;
  bool retained = false;

  try {
    const auto context_result = driver.retain_primary_context(0);
    if (context_result.status.result_code == CUDA_ERROR_NO_DEVICE) {
      std::cout << "CUDA device unavailable; skipping\n";
      return 77;
    }
    require_success(
        context_result.status,
        "retain primary context");
    retained = true;
    context = context_result.context;

    if (context == nullptr) {
      throw std::runtime_error(
          "successful context retain returned null");
    }
    if (context_result.compute_capability_major <= 0) {
      throw std::runtime_error(
          "invalid compute capability metadata");
    }

    const auto stream_result = driver.create_stream(
        context,
        CU_STREAM_NON_BLOCKING);
    require_success(stream_result.status, "create stream");
    stream = stream_result.stream;
    if (stream == nullptr) {
      throw std::runtime_error(
          "successful stream creation returned null");
    }
    require_success(
        driver.synchronize_stream(context, stream),
        "synchronize stream");
    require_success(
        driver.synchronize_context(context),
        "synchronize context");

    const auto event_result = driver.create_event(
        context,
        CU_EVENT_DISABLE_TIMING);
    require_success(event_result.status, "create event");
    event = event_result.event;
    if (event == nullptr) {
      throw std::runtime_error(
          "successful event creation returned null");
    }
    require_success(
        driver.record_event(context, event, stream),
        "record event");
    const auto query_result = driver.query_event(context, event);
    require_success(query_result.status, "query event");
    require_success(
        driver.wait_for_event(context, stream, event),
        "wait for event");
    require_success(
        driver.synchronize_event(context, event),
        "synchronize event");
    const auto completed_query = driver.query_event(context, event);
    require_success(completed_query.status, "query completed event");
    if (!completed_query.complete) {
      throw std::runtime_error(
          "synchronized event did not report completion");
    }

    {
      const std::vector<std::int32_t> pageable_source(8 * 1024 * 1024, 37);
      const auto byte_count = pageable_source.size() * sizeof(std::int32_t);
      const auto allocation = driver.allocate_device_memory(context, byte_count);
      require_success(allocation.status, "allocate pageable upload target");
      memory = allocation.address;
      flight4s::cuda::CurrentContextScope current(context);
      require_success(flight4s::cuda::make_driver_status(current.push_result()),
                      "enter upload test context");
      for (int iteration = 0; iteration < 32; ++iteration) {
        require_success(driver.copy_host_to_device(
                            context, memory, 0, pageable_source.data(), byte_count),
                        "synchronous pageable upload");
        // A successful synchronous upload must be ready for any subsequent stream.
        require_success(flight4s::cuda::make_driver_status(cuStreamQuery(CU_STREAM_LEGACY)),
                        "pageable upload completed before return");
      }
      std::vector<std::int32_t> copied(pageable_source.size());
      require_success(driver.copy_device_to_host(context, copied.data(), memory, 0, byte_count),
                      "read pageable upload target");
      if (copied != pageable_source) {
        throw std::runtime_error("pageable upload changed values");
      }
      require_success(flight4s::cuda::make_driver_status(current.close()),
                      "restore upload test context");
      require_success(driver.free_device_memory(context, memory), "free pageable upload target");
      memory = 0;
    }

    const std::array<std::int32_t, 4> host_source{
        11, 22, 33, 44};
    const auto pinned_result = driver.allocate_pinned_memory(
        context,
        sizeof(host_source));
    require_success(
        pinned_result.status,
        "allocate pinned memory");
    pinned_memory = pinned_result.address;
    if (pinned_memory == nullptr) {
      throw std::runtime_error(
          "successful pinned allocation returned null");
    }
    auto* pinned_values =
        static_cast<std::int32_t*>(pinned_memory);
    std::fill(pinned_values, pinned_values + host_source.size(), 0);
    const auto memory_result = driver.allocate_device_memory(
        context,
        sizeof(host_source));
    require_success(
        memory_result.status,
        "allocate device memory");
    memory = memory_result.address;
    if (memory == 0) {
      throw std::runtime_error(
          "successful device allocation returned null");
    }

    require_success(
        driver.copy_host_to_device(
            context,
            memory,
            0,
            pinned_memory,
            sizeof(host_source)),
        "initialize device memory");
    std::copy(host_source.begin(), host_source.end(), pinned_values);
    require_success(
        driver.copy_host_to_device_async(
            context,
            memory,
            sizeof(std::int32_t),
            pinned_values + 1,
            sizeof(std::int32_t) * 2,
            stream),
        "submit partial async host-to-device copy");
    require_success(
        driver.record_event(context, event, stream),
        "record async host-to-device completion");
    require_success(
        driver.synchronize_event(context, event),
        "synchronize async host-to-device completion");
    std::fill(pinned_values, pinned_values + host_source.size(), -1);
    require_success(
        driver.copy_device_to_host_async(
            context,
            pinned_values + 1,
            memory,
            sizeof(std::int32_t),
            sizeof(std::int32_t) * 2,
            stream),
        "submit partial async device-to-host copy");
    require_success(
        driver.record_event(context, event, stream),
        "record async device-to-host completion");
    require_success(
        driver.synchronize_event(context, event),
        "synchronize async device-to-host completion");
    const std::array<std::int32_t, 4> expected_partial{
        -1, 22, 33, -1};
    if (!std::equal(
            expected_partial.begin(),
            expected_partial.end(),
            pinned_values)) {
      throw std::runtime_error(
          "partial copy modified values outside the selected range");
    }

    const auto module_result =
        driver.load_ptx(context, read_file(arguments[1]));
    require_success(module_result.status, "load PTX module");
    module = module_result.module;
    if (module == nullptr) {
      throw std::runtime_error(
          "successful module load returned null");
    }

    const auto function_result = driver.resolve_function(
        context,
        module,
        "write_values");
    require_success(function_result.status, "resolve write_values");
    if (function_result.function == nullptr) {
      throw std::runtime_error(
          "successful function lookup returned null");
    }

    const auto attributes_result = driver.function_attributes(
        context,
        function_result.function);
    require_success(
        attributes_result.status,
        "query write_values attributes");
    if (attributes_result.attributes.max_threads_per_block <= 0 ||
        attributes_result.attributes.static_shared_memory_bytes < 0 ||
        attributes_result.attributes.constant_memory_bytes < 0 ||
        attributes_result.attributes.local_memory_bytes < 0 ||
        attributes_result.attributes.registers_per_thread < 0) {
      throw std::runtime_error(
          "CUDA function attributes contained an invalid value");
    }

    const auto missing_result = driver.resolve_function(
        context,
        module,
        "missing_kernel");
    if (missing_result.status.result_code != CUDA_ERROR_NOT_FOUND) {
      throw std::runtime_error(
          "missing function did not return CUDA_ERROR_NOT_FOUND");
    }

    const auto invalid_ptx = std::vector<std::uint8_t>{
        'n', 'o', 't', ' ', 'p', 't', 'x'};
    const auto invalid_result =
        driver.load_ptx(context, invalid_ptx);
    if (invalid_result.status.succeeded()) {
      throw std::runtime_error("invalid PTX unexpectedly loaded");
    }

    require_success(
        driver.unload_module(context, module),
        "unload module");
    module = nullptr;
    require_success(
        driver.free_device_memory(context, memory),
        "free device memory");
    memory = 0;
    require_success(
        driver.free_pinned_memory(context, pinned_memory),
        "free pinned memory");
    pinned_memory = nullptr;
    require_success(
        driver.destroy_event(context, event),
        "destroy event");
    event = nullptr;
    require_success(
        driver.destroy_stream(context, stream),
        "destroy stream");
    stream = nullptr;
    require_success(
        driver.release_primary_context(0),
        "release primary context");
    retained = false;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (memory != 0 && context != nullptr) {
      static_cast<void>(
          driver.free_device_memory(context, memory));
    }
    if (pinned_memory != nullptr && context != nullptr) {
      static_cast<void>(
          driver.free_pinned_memory(context, pinned_memory));
    }
    if (module != nullptr && context != nullptr) {
      static_cast<void>(driver.unload_module(context, module));
    }
    if (event != nullptr && context != nullptr) {
      static_cast<void>(driver.destroy_event(context, event));
    }
    if (stream != nullptr && context != nullptr) {
      static_cast<void>(driver.destroy_stream(context, stream));
    }
    if (retained) {
      static_cast<void>(driver.release_primary_context(0));
    }
    return 1;
  }

  std::cout << "CUDA Driver resource integration tests passed\n";
  return 0;
}
