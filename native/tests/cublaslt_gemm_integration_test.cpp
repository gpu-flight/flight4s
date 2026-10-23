#include <cmath>
#include <cstdio>
#include <vector>

#include "flight4s/cuda/cublaslt_gemm.hpp"
#include "flight4s/cuda/cuda_driver.hpp"

namespace {

constexpr int skip_exit_code = 77;

bool nearly_equal(float produced, double expected) {
  return std::abs(static_cast<double>(produced) - expected) <= 1e-4;
}

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

  const flight4s::cuda::CublasLtGemm blas;
  const auto handle_result = blas.create_handle(context);
  if (!handle_result.status.succeeded()) {
    std::printf(
        "handle creation failed: %s\n",
        handle_result.status.result_name.c_str());
    return 1;
  }

  // Row-major A (3 x 2), B (2 x 4), C (3 x 4); exact small-integer values.
  constexpr int m = 3;
  constexpr int n = 4;
  constexpr int k = 2;
  std::vector<float> a{1, 2, 3, 4, 5, 6};
  std::vector<float> b{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> c(static_cast<std::size_t>(m) * n, 1.0f);
  const float alpha = 2.0f;
  const float beta = 0.5f;

  const auto a_alloc = driver.allocate_device_memory(context, a.size() * 4);
  const auto b_alloc = driver.allocate_device_memory(context, b.size() * 4);
  const auto c_alloc = driver.allocate_device_memory(context, c.size() * 4);
  if (!a_alloc.status.succeeded() || !b_alloc.status.succeeded() ||
      !c_alloc.status.succeeded()) {
    std::printf("device allocation failed\n");
    return 1;
  }
  const auto upload_a =
      driver.copy_host_to_device(context, a_alloc.address, 0, a.data(), a.size() * 4);
  const auto upload_b =
      driver.copy_host_to_device(context, b_alloc.address, 0, b.data(), b.size() * 4);
  const auto upload_c =
      driver.copy_host_to_device(context, c_alloc.address, 0, c.data(), c.size() * 4);
  if (!upload_a.succeeded() || !upload_b.succeeded() || !upload_c.succeeded()) {
    std::printf("host-to-device upload failed\n");
    return 1;
  }

  const auto gemm_status = blas.execute_gemm_f32(
      context, handle_result.handle, nullptr,
      m, n, k, alpha, beta,
      a_alloc.address, b_alloc.address, c_alloc.address, 0, 0);
  if (!gemm_status.succeeded()) {
    std::printf("gemm failed: %s\n", gemm_status.result_name.c_str());
    return 1;
  }
  const auto synchronize_status = driver.synchronize_context(context);
  if (!synchronize_status.succeeded()) {
    std::printf("synchronize failed: %s\n", synchronize_status.result_name.c_str());
    return 1;
  }

  std::vector<float> produced(c.size());
  const auto download_status = driver.copy_device_to_host(
      context, produced.data(), c_alloc.address, 0, produced.size() * 4);
  if (!download_status.succeeded()) {
    std::printf("device-to-host download failed\n");
    return 1;
  }

  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double accumulated = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        accumulated += static_cast<double>(a[row * k + inner]) *
            static_cast<double>(b[inner * n + column]);
      }
      const double expected = alpha * accumulated + beta * 1.0;
      const float value = produced[row * n + column];
      if (!nearly_equal(value, expected)) {
        std::printf(
            "mismatch at (%d, %d): produced %f expected %f\n",
            row, column, static_cast<double>(value), expected);
        ++failures;
      }
    }
  }

  const auto free_a = driver.free_device_memory(context, a_alloc.address);
  const auto free_b = driver.free_device_memory(context, b_alloc.address);
  const auto free_c = driver.free_device_memory(context, c_alloc.address);
  if (!free_a.succeeded() || !free_b.succeeded() || !free_c.succeeded()) {
    std::printf("device free failed\n");
    ++failures;
  }

  const auto destroy_status = blas.destroy_handle(context, handle_result.handle);
  if (!destroy_status.succeeded()) {
    std::printf(
        "handle destruction failed: %s\n",
        destroy_status.result_name.c_str());
    ++failures;
  }
  const auto release_status = driver.release_primary_context(0);
  if (!release_status.succeeded()) {
    std::printf("context release failed\n");
    ++failures;
  }

  if (failures == 0) {
    std::printf("cublaslt gemm integration: all checks passed\n");
    return 0;
  }
  return 1;
}
