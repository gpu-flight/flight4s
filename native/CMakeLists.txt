cmake_minimum_required(VERSION 3.24)

project(flight4s_cuda_native LANGUAGES CXX)

if(MSVC)
  set(
    CMAKE_MSVC_RUNTIME_LIBRARY
    "MultiThreaded$<$<CONFIG:Debug>:Debug>"
  )
endif()

option(
  FLIGHT4S_BUILD_CUDA_INTEGRATION_TESTS
  "Build tests that require a CUDA-capable device"
  OFF
)

find_package(JNI REQUIRED)
find_package(CUDAToolkit 12.0 REQUIRED)

add_library(
  flight4s_launch_contract
  STATIC
  src/launch_contract.cpp
)
target_compile_features(flight4s_launch_contract PUBLIC cxx_std_20)
target_include_directories(
  flight4s_launch_contract
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
set_target_properties(
  flight4s_launch_contract
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_library(
  flight4s_nvrtc_compiler
  STATIC
  src/nvrtc_compiler.cpp
)
target_compile_features(flight4s_nvrtc_compiler PUBLIC cxx_std_20)
target_include_directories(
  flight4s_nvrtc_compiler
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(
  flight4s_nvrtc_compiler
  PUBLIC
  CUDA::nvrtc
)
set_target_properties(
  flight4s_nvrtc_compiler
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_library(
  flight4s_cuda_driver
  STATIC
  src/current_context_scope.cpp
  src/cuda_driver.cpp
)
target_compile_features(flight4s_cuda_driver PUBLIC cxx_std_20)
target_include_directories(
  flight4s_cuda_driver
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(
  flight4s_cuda_driver
  PUBLIC
  CUDA::cuda_driver
)
set_target_properties(
  flight4s_cuda_driver
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_library(
  flight4s_cublaslt
  STATIC
  src/cublaslt_gemm.cpp
)
target_compile_features(flight4s_cublaslt PUBLIC cxx_std_20)
target_include_directories(
  flight4s_cublaslt
  PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(
  flight4s_cublaslt
  PUBLIC
  CUDA::cublasLt
  CUDA::cuda_driver
)
set_target_properties(
  flight4s_cublaslt
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)

add_library(
  flight4s_cuda
  SHARED
  src/cuda_launcher.cpp
  src/jni_cuda_result.cpp
  src/jni_direct_buffer.cpp
  src/jni_environment.cpp
  src/jni_cuda_launcher.cpp
  src/jni_cuda_driver.cpp
  src/jni_nvrtc_compiler.cpp
  src/jni_cublaslt.cpp
)
target_compile_features(flight4s_cuda PRIVATE cxx_std_20)
target_include_directories(
  flight4s_cuda
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  ${JNI_INCLUDE_DIRS}
)
target_link_libraries(
  flight4s_cuda
  PRIVATE
  flight4s_launch_contract
  flight4s_cuda_driver
  flight4s_nvrtc_compiler
  flight4s_cublaslt
  CUDA::cuda_driver
)

if(MSVC)
  target_compile_options(
    flight4s_launch_contract
    PRIVATE
    /W4
    /permissive-
  )
  target_compile_options(
    flight4s_nvrtc_compiler
    PRIVATE
    /W4
    /permissive-
  )
  target_compile_options(
    flight4s_cuda_driver
    PRIVATE
    /W4
    /permissive-
  )
  target_compile_options(
    flight4s_cuda
    PRIVATE
    /W4
    /permissive-
  )
else()
  target_compile_options(
    flight4s_launch_contract
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_compile_options(
    flight4s_nvrtc_compiler
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_compile_options(
    flight4s_cuda_driver
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
  target_compile_options(
    flight4s_cuda
    PRIVATE
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

include(CTest)

if(BUILD_TESTING)
  add_executable(
    flight4s_launch_contract_tests
    tests/launch_contract_test.cpp
  )
  target_compile_features(
    flight4s_launch_contract_tests
    PRIVATE
    cxx_std_20
  )
  target_link_libraries(
    flight4s_launch_contract_tests
    PRIVATE
    flight4s_launch_contract
  )
  add_test(
    NAME flight4s_launch_contract
    COMMAND flight4s_launch_contract_tests
  )

  add_executable(
    flight4s_nvrtc_compiler_tests
    tests/nvrtc_compiler_test.cpp
  )
  target_compile_features(
    flight4s_nvrtc_compiler_tests
    PRIVATE
    cxx_std_20
  )
  target_link_libraries(
    flight4s_nvrtc_compiler_tests
    PRIVATE
    flight4s_nvrtc_compiler
  )
  if(MSVC)
    target_compile_options(
      flight4s_nvrtc_compiler_tests
      PRIVATE
      /W4
      /permissive-
    )
  else()
    target_compile_options(
      flight4s_nvrtc_compiler_tests
      PRIVATE
      -Wall
      -Wextra
      -Wpedantic
    )
  endif()
  add_test(
    NAME flight4s_nvrtc_compiler
    COMMAND flight4s_nvrtc_compiler_tests
  )

  if(FLIGHT4S_BUILD_CUDA_INTEGRATION_TESTS)
    set(
      FLIGHT4S_TEST_PTX
      "${CMAKE_CURRENT_BINARY_DIR}/flight4s_test_kernel.ptx"
    )
    add_custom_command(
      OUTPUT "${FLIGHT4S_TEST_PTX}"
      COMMAND
        "${CUDAToolkit_NVCC_EXECUTABLE}"
        --ptx
        --std=c++20
        --gpu-architecture=compute_90
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kernel.cu"
        --output-file
        "${FLIGHT4S_TEST_PTX}"
      DEPENDS tests/test_kernel.cu
      VERBATIM
    )
    add_custom_target(
      flight4s_test_ptx
      DEPENDS "${FLIGHT4S_TEST_PTX}"
    )

    add_executable(
      flight4s_cuda_launch_integration_tests
      tests/cuda_launch_integration_test.cpp
      src/cuda_launcher.cpp
    )
    add_dependencies(
      flight4s_cuda_launch_integration_tests
      flight4s_test_ptx
    )
    target_compile_features(
      flight4s_cuda_launch_integration_tests
      PRIVATE
      cxx_std_20
    )
    target_include_directories(
      flight4s_cuda_launch_integration_tests
      PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_link_libraries(
      flight4s_cuda_launch_integration_tests
      PRIVATE
      flight4s_cuda_driver
      flight4s_launch_contract
      CUDA::cuda_driver
    )
    if(MSVC)
      target_compile_options(
        flight4s_cuda_launch_integration_tests
        PRIVATE
        /W4
        /permissive-
      )
    else()
      target_compile_options(
        flight4s_cuda_launch_integration_tests
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
      )
    endif()

    add_test(
      NAME flight4s_cuda_launch_integration
      COMMAND
        flight4s_cuda_launch_integration_tests
        "${FLIGHT4S_TEST_PTX}"
    )
    set_tests_properties(
      flight4s_cuda_launch_integration
      PROPERTIES
      LABELS gpu
      SKIP_RETURN_CODE 77
    )

    add_executable(
      flight4s_cuda_driver_integration_tests
      tests/cuda_driver_integration_test.cpp
    )
    add_dependencies(
      flight4s_cuda_driver_integration_tests
      flight4s_test_ptx
    )
    target_compile_features(
      flight4s_cuda_driver_integration_tests
      PRIVATE
      cxx_std_20
    )
    target_link_libraries(
      flight4s_cuda_driver_integration_tests
      PRIVATE
      flight4s_cuda_driver
    )
    if(MSVC)
      target_compile_options(
        flight4s_cuda_driver_integration_tests
        PRIVATE
        /W4
        /permissive-
      )
    else()
      target_compile_options(
        flight4s_cuda_driver_integration_tests
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
      )
    endif()

    add_test(
      NAME flight4s_cuda_driver_integration
      COMMAND
        flight4s_cuda_driver_integration_tests
        "${FLIGHT4S_TEST_PTX}"
    )
    set_tests_properties(
      flight4s_cuda_driver_integration
      PROPERTIES
      LABELS gpu
      SKIP_RETURN_CODE 77
    )

    add_executable(
      flight4s_cublaslt_gemm_integration_tests
      tests/cublaslt_gemm_integration_test.cpp
    )
    target_compile_features(
      flight4s_cublaslt_gemm_integration_tests
      PRIVATE
      cxx_std_20
    )
    target_link_libraries(
      flight4s_cublaslt_gemm_integration_tests
      PRIVATE
      flight4s_cublaslt
      flight4s_cuda_driver
    )
    if(MSVC)
      target_compile_options(
        flight4s_cublaslt_gemm_integration_tests
        PRIVATE
        /W4
        /permissive-
      )
    else()
      target_compile_options(
        flight4s_cublaslt_gemm_integration_tests
        PRIVATE
        -Wall
        -Wextra
        -Wpedantic
      )
    endif()

    add_test(
      NAME flight4s_cublaslt_gemm_integration
      COMMAND flight4s_cublaslt_gemm_integration_tests
    )
    set_tests_properties(
      flight4s_cublaslt_gemm_integration
      PROPERTIES
      LABELS gpu
      SKIP_RETURN_CODE 77
    )
  endif()
endif()
