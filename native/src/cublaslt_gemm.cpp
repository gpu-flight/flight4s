#include "flight4s/cuda/cublaslt_gemm.hpp"

#include <stdexcept>
#include <string>

#include "flight4s/cuda/current_context_scope.hpp"

namespace flight4s::cuda {

namespace {

// cublasStatus_t shares CUBLAS_STATUS_SUCCESS == 0 with CUDA_SUCCESS, so the
// existing status plumbing carries blas results with blas-specific names.
std::string blas_status_name(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "CUBLAS_STATUS_UNKNOWN";
  }
}

CudaDriverStatus make_blas_status(
    cublasStatus_t status,
    const std::string& operation) {
  return CudaDriverStatus{
      static_cast<CUresult>(status),
      blas_status_name(status),
      status == CUBLAS_STATUS_SUCCESS
          ? std::string{"no error"}
          : operation + " failed",
      {},
      {}};
}

CudaDriverStatus blas_scope_failure(CUresult push_result) {
  return make_driver_status(push_result);
}

void require_context(CUcontext context) {
  if (context == nullptr) {
    throw std::invalid_argument("CUDA context handle must not be null");
  }
}

}  // namespace

CublasLtHandleResult CublasLtGemm::create_handle(CUcontext context) const {
  require_context(context);
  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return {blas_scope_failure(current.push_result()), nullptr};
  }
  cublasLtHandle_t handle = nullptr;
  const auto status = cublasLtCreate(&handle);
  const auto pop_result = current.close();
  if (status != CUBLAS_STATUS_SUCCESS) {
    return {make_blas_status(status, "cuBLASLt handle creation"), nullptr};
  }
  if (pop_result != CUDA_SUCCESS) {
    return {blas_scope_failure(pop_result), nullptr};
  }
  return {make_blas_status(status, "cuBLASLt handle creation"), handle};
}

CudaDriverStatus CublasLtGemm::destroy_handle(
    CUcontext context,
    cublasLtHandle_t handle) const {
  require_context(context);
  if (handle == nullptr) {
    throw std::invalid_argument("cuBLASLt handle must not be null");
  }
  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return blas_scope_failure(current.push_result());
  }
  const auto status = cublasLtDestroy(handle);
  const auto pop_result = current.close();
  if (status != CUBLAS_STATUS_SUCCESS) {
    return make_blas_status(status, "cuBLASLt handle destruction");
  }
  if (pop_result != CUDA_SUCCESS) {
    return blas_scope_failure(pop_result);
  }
  return make_blas_status(status, "cuBLASLt handle destruction");
}

CudaDriverStatus CublasLtGemm::execute_gemm_f32(
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
    std::size_t workspace_bytes) const {
  require_context(context);
  if (handle == nullptr) {
    throw std::invalid_argument("cuBLASLt handle must not be null");
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    throw std::invalid_argument("GEMM dimensions must be positive");
  }
  if (a == 0 || b == 0 || c == 0) {
    throw std::invalid_argument("GEMM device pointers must not be null");
  }
  if (workspace == 0 && workspace_bytes != 0) {
    throw std::invalid_argument(
        "GEMM workspace bytes require a workspace pointer");
  }

  CurrentContextScope current(context);
  if (current.push_result() != CUDA_SUCCESS) {
    return blas_scope_failure(current.push_result());
  }

  cublasLtMatmulDesc_t matmul = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  auto status = CUBLAS_STATUS_SUCCESS;
  std::string operation = "cuBLASLt matmul";

  const auto row_order = CUBLASLT_ORDER_ROW;
  auto configure_layout = [&](cublasLtMatrixLayout_t* layout,
                              std::int64_t rows,
                              std::int64_t columns,
                              std::int64_t leading) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      return;
    }
    status = cublasLtMatrixLayoutCreate(
        layout, CUDA_R_32F, rows, columns, leading);
    if (status != CUBLAS_STATUS_SUCCESS) {
      operation = "cuBLASLt matrix layout creation";
      return;
    }
    status = cublasLtMatrixLayoutSetAttribute(
        *layout,
        CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order,
        sizeof(row_order));
    if (status != CUBLAS_STATUS_SUCCESS) {
      operation = "cuBLASLt matrix layout order";
    }
  };

  // Full FP32 accumulation; TF32 downconversion is deliberately not reachable.
  status = cublasLtMatmulDescCreate(&matmul, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) {
    operation = "cuBLASLt matmul descriptor creation";
  }
  configure_layout(&layout_a, m, k, k);
  configure_layout(&layout_b, k, n, n);
  configure_layout(&layout_c, m, n, n);

  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmul(
        handle,
        matmul,
        &alpha,
        reinterpret_cast<const void*>(a),
        layout_a,
        reinterpret_cast<const void*>(b),
        layout_b,
        &beta,
        reinterpret_cast<const void*>(c),
        layout_c,
        reinterpret_cast<void*>(c),
        layout_c,
        nullptr,
        reinterpret_cast<void*>(workspace),
        workspace_bytes,
        stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
      operation = "cuBLASLt matmul";
    }
  }

  if (layout_c != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_c);
  }
  if (layout_b != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_b);
  }
  if (layout_a != nullptr) {
    cublasLtMatrixLayoutDestroy(layout_a);
  }
  if (matmul != nullptr) {
    cublasLtMatmulDescDestroy(matmul);
  }

  const auto pop_result = current.close();
  if (status != CUBLAS_STATUS_SUCCESS) {
    return make_blas_status(status, operation);
  }
  if (pop_result != CUDA_SUCCESS) {
    return blas_scope_failure(pop_result);
  }
  return make_blas_status(status, operation);
}

}  // namespace flight4s::cuda
