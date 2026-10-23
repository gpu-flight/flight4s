#include <jni.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>

#include "flight4s/cuda/cublaslt_gemm.hpp"
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

class CublasLtJniAdapter final {
 public:
  explicit CublasLtJniAdapter(JNIEnv* environment) noexcept
      : results_(environment) {}

  [[nodiscard]] jobject create_handle(jlong context_handle) const {
    const auto result = blas_.create_handle(
        from_java_handle<CUcontext>(context_handle));
    return results_.driver_result(
        result.status,
        to_java_handle(result.handle));
  }

  [[nodiscard]] jobject destroy_handle(
      jlong context_handle,
      jlong blas_handle) const {
    const auto status = blas_.destroy_handle(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<cublasLtHandle_t>(blas_handle));
    return results_.driver_result(status);
  }

  [[nodiscard]] jobject execute_gemm_f32(
      jlong context_handle,
      jlong blas_handle,
      jlong stream_handle,
      jint m,
      jint n,
      jint k,
      jfloat alpha,
      jfloat beta,
      jlong a_address,
      jlong b_address,
      jlong c_address,
      jlong workspace_address,
      jlong workspace_bytes) const {
    if (workspace_bytes < 0) {
      throw std::invalid_argument(
          "GEMM workspace size must not be negative");
    }
    const auto status = blas_.execute_gemm_f32(
        from_java_handle<CUcontext>(context_handle),
        from_java_handle<cublasLtHandle_t>(blas_handle),
        from_java_handle<CUstream>(stream_handle),
        m,
        n,
        k,
        alpha,
        beta,
        from_java_device_address(a_address),
        from_java_device_address(b_address),
        from_java_device_address(c_address),
        from_java_device_address(workspace_address),
        static_cast<std::size_t>(workspace_bytes));
    return results_.driver_result(status);
  }

 private:
  flight4s::cuda::CublasLtGemm blas_;
  flight4s::jni::CudaJniResultFactory results_;
};

}  // namespace

extern "C" JNIEXPORT jobject JNICALL
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_createBlasHandle(
    JNIEnv* environment,
    jclass,
    jlong context_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CublasLtJniAdapter(environment).create_handle(context_handle);
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
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_destroyBlasHandle(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong blas_handle) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CublasLtJniAdapter(environment)
        .destroy_handle(context_handle, blas_handle);
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
Java_flight4s_runtime_cuda_internal_CudaNativeBindings_executeGemmF32(
    JNIEnv* environment,
    jclass,
    jlong context_handle,
    jlong blas_handle,
    jlong stream_handle,
    jint m,
    jint n,
    jint k,
    jfloat alpha,
    jfloat beta,
    jlong a_address,
    jlong b_address,
    jlong c_address,
    jlong workspace_address,
    jlong workspace_bytes) {
  const flight4s::jni::JniEnvironment jni(environment);
  try {
    return CublasLtJniAdapter(environment).execute_gemm_f32(
        context_handle,
        blas_handle,
        stream_handle,
        m,
        n,
        k,
        alpha,
        beta,
        a_address,
        b_address,
        c_address,
        workspace_address,
        workspace_bytes);
  } catch (const std::invalid_argument& error) {
    jni.throw_illegal_argument(error.what());
  } catch (const std::bad_alloc& error) {
    jni.throw_out_of_memory(error.what());
  } catch (const std::exception& error) {
    jni.throw_runtime_exception(error.what());
  }
  return nullptr;
}
