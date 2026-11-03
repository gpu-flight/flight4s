#include <jni.h>

#include <cstdint>
#include <exception>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "flight4s/cuda/cuda_driver.hpp"
#include "jni_direct_buffer.hpp"
#include "jni_cuda_result.hpp"
#include "jni_environment.hpp"

namespace {

template <typename Handle>
Handle from_java_handle(jlong handle) {
  return reinterpret_cast<Handle>(
      static_cast<std::uintptr_t>(handle));
}

template <typename Handle>
jlong to_java_handle(Handle handle) {
  return static_cast<jlong>(
      reinterpret_cast<std::uintptr_t>(handle));
}

CUdeviceptr from_java_device_address(jlong address) {
  return static_cast<CUdeviceptr>(
      static_cast<std::uint64_t>(address));
}

jlong to_java_device_address(CUdeviceptr address) {
  return static_cast<jlong>(address);
}

class CudaDriverJniAdapter final {
 public:
  explicit CudaDriverJniAdapter(JNIEnv* environment) noexcept
      : environment_(environment),
        results_(environment) {}

  [[nodiscard]] jobject retain_primary_context(
      jint device_ordinal) const {
    const auto result =
        driver_.retain_primary_context(device_ordinal);
    return results_.context_result(result);
  }

  [[nodiscard]] jobject release_primary_context(
      jint device_ordinal) const {
    const auto status =
        driver_.release_primary_context(device_ordinal);
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject synchronize_context(
      jlong context_handle) const {
    const auto status = driver_.synchronize_context(
        from_java_handle<CUcontext>(context_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject load_ptx(
      jlong context_handle,
      jbyteArray ptx) const {
    const auto bytes = read_bytes(ptx, "PTX");
    if (has_exception()) {
      return nullptr;
    }
    const auto result = driver_.load_ptx(
        from_java_handle<CUcontext>(context_handle),
        bytes);
    return results_.driver_result(
        result.status,
        to_java_handle(result.module));
  }

  [[nodiscard]] jobject unload_module(
      jlong context_handle,
      jlong module_handle) const {
    const auto status = driver_.unload_module(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUmodule>(module_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject resolve_function(
      jlong context_handle,
      jlong module_handle,
      jbyteArray function_name_utf8) const {
    const auto name_bytes =
        read_bytes(function_name_utf8, "CUDA function name");
    if (has_exception()) {
      return nullptr;
    }
    const std::string name(name_bytes.begin(), name_bytes.end());
    const auto result = driver_.resolve_function(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUmodule>(module_handle),
        name);
    return results_.driver_result(
        result.status,
        to_java_handle(result.function));
  }

  [[nodiscard]] jobject query_function_attributes(
      jlong context_handle,
      jlong function_handle) const {
    const auto result = driver_.function_attributes(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUfunction>(function_handle));
    return results_.function_attributes_result(result);
  }

  [[nodiscard]] jobject create_stream(
      jlong context_handle,
      jint flags) const {
    if (flags != CU_STREAM_DEFAULT &&
        flags != CU_STREAM_NON_BLOCKING) {
      throw std::invalid_argument(
          "unsupported CUDA stream flags");
    }
    const auto result = driver_.create_stream(
        from_java_handle<CUcontext>(context_handle),
        static_cast<std::uint32_t>(flags));
    return results_.driver_result(
        result.status,
        to_java_handle(result.stream));
  }

  [[nodiscard]] jobject destroy_stream(
      jlong context_handle,
      jlong stream_handle) const {
    const auto status = driver_.destroy_stream(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject synchronize_stream(
      jlong context_handle,
      jlong stream_handle) const {
    const auto status = driver_.synchronize_stream(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject create_event(
      jlong context_handle,
      jint flags) const {
    constexpr jint supported_flags =
        CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING;
    if ((flags & ~supported_flags) != 0) {
      throw std::invalid_argument(
          "unsupported CUDA event flags");
    }
    const auto result = driver_.create_event(
        from_java_handle<CUcontext>(context_handle),
        static_cast<std::uint32_t>(flags));
    return results_.driver_result(
        result.status,
        to_java_handle(result.event));
  }

  [[nodiscard]] jobject destroy_event(
      jlong context_handle,
      jlong event_handle) const {
    const auto status = driver_.destroy_event(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUevent>(event_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject record_event(
      jlong context_handle,
      jlong event_handle,
      jlong stream_handle) const {
    const auto status = driver_.record_event(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUevent>(event_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject query_event(
      jlong context_handle,
      jlong event_handle) const {
    const auto result = driver_.query_event(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUevent>(event_handle));
    return results_.event_query_result(result);
  }

  [[nodiscard]] jobject synchronize_event(
      jlong context_handle,
      jlong event_handle) const {
    const auto status = driver_.synchronize_event(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUevent>(event_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject wait_for_event(
      jlong context_handle,
      jlong stream_handle,
      jlong event_handle) const {
    const auto status = driver_.wait_for_event(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUstream>(stream_handle),
        from_java_handle<CUevent>(event_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject allocate_pinned_memory(
      jlong context_handle,
      jlong size_bytes) const {
    if (size_bytes <= 0 ||
        size_bytes > std::numeric_limits<jint>::max()) {
      throw std::invalid_argument(
          "CUDA pinned allocation size must be positive and fit "
          "in a JVM direct buffer");
    }
    const auto context =
        from_java_handle<CUcontext>(context_handle);
    const auto result = driver_.allocate_pinned_memory(
        context,
        static_cast<std::uint64_t>(size_bytes));
    try {
      auto java_result =
          results_.pinned_memory_result(result, size_bytes);
      if (java_result == nullptr && result.address != nullptr) {
        release_pinned_after_jni_failure(context, result.address);
      }
      return java_result;
    } catch (...) {
      if (result.address != nullptr) {
        release_pinned_after_jni_failure(context, result.address);
      }
      throw;
    }
  }

  [[nodiscard]] jobject free_pinned_memory(
      jlong context_handle,
      jlong host_address) const {
    const auto status = driver_.free_pinned_memory(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<void*>(host_address));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject begin_stream_capture(
      jlong context_handle,
      jlong stream_handle) const {
    const auto status = driver_.begin_stream_capture(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject end_stream_capture(
      jlong context_handle,
      jlong stream_handle) const {
    const auto result = driver_.end_stream_capture(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(
        result.status,
        to_java_handle(result.graph));
  }

  [[nodiscard]] jobject instantiate_graph(
      jlong context_handle,
      jlong graph_handle) const {
    const auto result = driver_.instantiate_graph(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUgraph>(graph_handle));
    return results_.driver_result(
        result.status,
        to_java_handle(result.exec));
  }

  [[nodiscard]] jobject launch_graph(
      jlong context_handle,
      jlong exec_handle,
      jlong stream_handle) const {
    const auto status = driver_.launch_graph(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUgraphExec>(exec_handle),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject destroy_graph_exec(
      jlong context_handle,
      jlong exec_handle) const {
    const auto status = driver_.destroy_graph_exec(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUgraphExec>(exec_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject destroy_graph(
      jlong context_handle,
      jlong graph_handle) const {
    const auto status = driver_.destroy_graph(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<CUgraph>(graph_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject allocate_device_memory(
      jlong context_handle,
      jlong size_bytes) const {
    if (size_bytes <= 0) {
      throw std::invalid_argument(
          "CUDA allocation size must be positive");
    }
    const auto result = driver_.allocate_device_memory(
        from_java_handle<CUcontext>(context_handle),
        static_cast<std::uint64_t>(size_bytes));
    return results_.driver_result(
        result.status,
        to_java_device_address(result.address));
  }

  [[nodiscard]] jobject free_device_memory(
      jlong context_handle,
      jlong device_address) const {
    const auto status = driver_.free_device_memory(
        from_java_handle<CUcontext>(context_handle),
        from_java_device_address(device_address));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject copy_host_to_device(
      jlong context_handle,
      jlong device_address,
      jlong device_offset_bytes,
      jobject source) const {
    if (device_offset_bytes < 0) {
      throw std::invalid_argument(
          "CUDA destination offset must not be negative");
    }
    const auto view = flight4s::jni::DirectBufferView::require_exact(
        environment_,
        source,
        "host copy source");
    if (has_exception()) {
      return nullptr;
    }
    const auto status = driver_.copy_host_to_device(
        from_java_handle<CUcontext>(context_handle),
        from_java_device_address(device_address),
        static_cast<std::uint64_t>(device_offset_bytes),
        view.data(),
        static_cast<std::uint64_t>(view.size_bytes()));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject copy_host_to_device_async(
      jlong context_handle,
      jlong device_address,
      jlong device_offset_bytes,
      jobject source,
      jlong stream_handle) const {
    if (device_offset_bytes < 0) {
      throw std::invalid_argument(
          "CUDA destination offset must not be negative");
    }
    const auto view = flight4s::jni::DirectBufferView::require_exact(
        environment_,
        source,
        "async host copy source");
    if (has_exception()) {
      return nullptr;
    }
    const auto status = driver_.copy_host_to_device_async(
        from_java_handle<CUcontext>(context_handle),
        from_java_device_address(device_address),
        static_cast<std::uint64_t>(device_offset_bytes),
        view.data(),
        static_cast<std::uint64_t>(view.size_bytes()),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject copy_device_to_host(
      jlong context_handle,
      jlong device_address,
      jlong device_offset_bytes,
      jobject destination) const {
    if (device_offset_bytes < 0) {
      throw std::invalid_argument(
          "CUDA source offset must not be negative");
    }
    const auto view = flight4s::jni::DirectBufferView::require_exact(
        environment_,
        destination,
        "host copy destination");
    if (has_exception()) {
      return nullptr;
    }
    const auto status = driver_.copy_device_to_host(
        from_java_handle<CUcontext>(context_handle),
        view.data(),
        from_java_device_address(device_address),
        static_cast<std::uint64_t>(device_offset_bytes),
        static_cast<std::uint64_t>(view.size_bytes()));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject copy_device_to_host_async(
      jlong context_handle,
      jlong device_address,
      jlong device_offset_bytes,
      jobject destination,
      jlong stream_handle) const {
    if (device_offset_bytes < 0) {
      throw std::invalid_argument(
          "CUDA source offset must not be negative");
    }
    const auto view = flight4s::jni::DirectBufferView::require_exact(
        environment_,
        destination,
        "async host copy destination");
    if (has_exception()) {
      return nullptr;
    }
    const auto status = driver_.copy_device_to_host_async(
        from_java_handle<CUcontext>(context_handle),
        view.data(),
        from_java_device_address(device_address),
        static_cast<std::uint64_t>(device_offset_bytes),
        static_cast<std::uint64_t>(view.size_bytes()),
        from_java_handle<CUstream>(stream_handle));
    return results_.driver_result(status);
  }

 private:
  [[nodiscard]] bool has_exception() const noexcept {
    return environment_->ExceptionCheck() == JNI_TRUE;
  }

  void release_pinned_after_jni_failure(
      CUcontext context,
      void* address) const noexcept {
    try {
      static_cast<void>(driver_.free_pinned_memory(context, address));
    } catch (...) {
    }
  }

  [[nodiscard]] std::vector<std::uint8_t> read_bytes(
      jbyteArray bytes,
      const char* description) const {
    if (bytes == nullptr) {
      throw std::invalid_argument(
          std::string(description) + " must not be null");
    }

    const jsize size = environment_->GetArrayLength(bytes);
    std::vector<std::uint8_t> value(
        static_cast<std::size_t>(size));
    if (size > 0) {
      environment_->GetByteArrayRegion(
          bytes,
          0,
          size,
          reinterpret_cast<jbyte*>(value.data()));
    }
    return value;
  }

  JNIEnv* environment_;
  flight4s::jni::CudaJniResultFactory results_;
  flight4s::cuda::CudaDriver driver_;
};

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_retainPrimaryContext(
    JNIEnv* environment,
    jclass,
    jint device_ordinal) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .retain_primary_context(device_ordinal);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_releasePrimaryContext(
    JNIEnv* environment,
    jclass,
    jint device_ordinal) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .release_primary_context(device_ordinal);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_synchronizeContext(
    JNIEnv* environment,
    jclass,
    jlong context_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .synchronize_context(context_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_loadPtx(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jbyteArray ptx) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .load_ptx(context_handle, ptx);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_unloadModule(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong module_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .unload_module(context_handle, module_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_resolveFunction(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong module_handle,
    jbyteArray function_name_utf8) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .resolve_function(
            context_handle,
            module_handle,
            function_name_utf8);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_queryFunctionAttributes(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong function_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .query_function_attributes(context_handle, function_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_createStream(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jint flags) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .create_stream(context_handle, flags);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_destroyStream(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .destroy_stream(context_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_synchronizeStream(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .synchronize_stream(context_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_createEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jint flags) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .create_event(context_handle, flags);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_destroyEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong event_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .destroy_event(context_handle, event_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_recordEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong event_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .record_event(context_handle, event_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_queryEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong event_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .query_event(context_handle, event_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_synchronizeEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong event_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .synchronize_event(context_handle, event_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_waitForEvent(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong stream_handle,
    jlong event_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .wait_for_event(context_handle, stream_handle, event_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_allocatePinnedMemory(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong size_bytes) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .allocate_pinned_memory(context_handle, size_bytes);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_freePinnedMemory(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong host_address) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .free_pinned_memory(context_handle, host_address);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_beginStreamCapture(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .begin_stream_capture(context_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_endStreamCapture(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .end_stream_capture(context_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_instantiateGraph(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong graph_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .instantiate_graph(context_handle, graph_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_launchGraph(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong exec_handle,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .launch_graph(context_handle, exec_handle, stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_destroyGraphExec(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong exec_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .destroy_graph_exec(context_handle, exec_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_destroyGraph(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong graph_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .destroy_graph(context_handle, graph_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_allocateDeviceMemory(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong size_bytes) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .allocate_device_memory(context_handle, size_bytes);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_freeDeviceMemory(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong device_address) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .free_device_memory(context_handle, device_address);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_copyHostToDevice(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong device_address,
    jlong device_offset_bytes,
    jobject source) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .copy_host_to_device(
            context_handle,
            device_address,
            device_offset_bytes,
            source);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_copyHostToDeviceAsync(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong device_address,
    jlong device_offset_bytes,
    jobject source,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .copy_host_to_device_async(
            context_handle,
            device_address,
            device_offset_bytes,
            source,
            stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_copyDeviceToHost(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong device_address,
    jlong device_offset_bytes,
    jobject destination) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .copy_device_to_host(
            context_handle,
            device_address,
            device_offset_bytes,
            destination);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_copyDeviceToHostAsync(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong device_address,
    jlong device_offset_bytes,
    jobject destination,
    jlong stream_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CudaDriverJniAdapter(environment)
        .copy_device_to_host_async(
            context_handle,
            device_address,
            device_offset_bytes,
            destination,
            stream_handle);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}
