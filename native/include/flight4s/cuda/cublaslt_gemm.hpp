#pragma once

#include <cublasLt.h>
#include <cuda.h>

#include <cstddef>
#include <cstdint>

#include "flight4s/cuda/cuda_driver.hpp"

namespace flight4s::cuda {

struct CublasLtHandleResult {
  CudaDriverStatus status;
  cublasLtHandle_t handle;
};

class CublasLtGemm final {
 public:
  [[nodiscard]] CublasLtHandleResult create_handle(CUcontext context) const;

  [[nodiscard]] CudaDriverStatus destroy_handle(
      CUcontext context,
      cublasLtHandle_t handle) const;

  // Row-major C = alpha * A x B + beta * C with full FP32 compute, never TF32.
  [[nodiscard]] CudaDriverStatus execute_gemm_f32(
      CUcontext context,
      cublasLtHandle_t handle,
      CUstream stream,
      std::int32_t m,
      std::int32_t n,
      std::int32_t k,
      float alpha,
      float beta,
      CUdeviceptr a,
      CUdeviceptr b,
      CUdeviceptr c,
      CUdeviceptr workspace,
      std::size_t workspace_bytes) const;
};

}  // namespace flight4s::cuda
