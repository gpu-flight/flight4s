# Flight4s CUDA Native Runtime

This directory builds the internal JNI library for three coarse native
operations:

- compile one CUDA C++ source artifact to PTX with NVRTC;
- retain CUDA primary contexts, load PTX modules, resolve functions, and unload
  resources through the CUDA Driver API;
- allocate and free device memory and perform synchronous or explicit-stream
  asynchronous whole-buffer and partial-range host/device copies;
- allocate page-locked host memory, expose an exact JNI direct-buffer view, and
  release the native allocation;
- create, record, query, synchronize, wait on, and destroy CUDA events;
- synchronize CUDA contexts and create, synchronize, and destroy explicit CUDA
  streams;
- validate one typed Flight4s launch request, construct CUDA's `void**` kernel
  parameter table, establish the owning context, and call `cuLaunchKernelEx`.

The launcher restores the caller's previous current context after submission
and does not synchronize the CUDA context or stream. Scala runtime objects own
retained primary contexts, loaded modules, device allocations, and explicit
streams and events. They also own page-locked host allocations surfaced as
internal direct buffers. CUDA requires participating resources to remain alive
until completion. The Scala runtime enforces that lifetime through deferred
native release. Explicit-stream work completes through stream, event, or
context synchronization; default-stream launches complete through context
synchronization. Closing a stream or context drains the tracked work owned by
that completion boundary before native destruction.

## Requirements

- CMake 3.24 or newer
- a C++20 compiler (GCC 10+, Clang 10+, or Visual Studio 2022+)
- JDK headers
- CUDA Toolkit 12.0 or newer
- an NVIDIA driver compatible with the selected toolkit

## Build And Contract Tests

```shell
cmake -S native -B native/build -DBUILD_TESTING=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

`flight4s_launch_contract` does not require a GPU. It verifies launch geometry,
stable descriptor codes, argument alignment and bounds, and `void**` pointer
construction.

`flight4s_nvrtc_compiler` also does not require a GPU. It verifies successful
CUDA C++ compilation, PTX retrieval, compiler-version metadata, complete
failure logs, and request validation.

## GPU Integration Tests

The optional integration tests compile a small PTX kernel, exercise primary
context/module/function, stream/event, device-memory, and pinned-host-memory
ownership, validate event completion and stream waits, round-trip bytes through
synchronous copies, and execute ordinary and clustered launches through
`cuLaunchKernelEx` while verifying context restoration. They require a CUDA
device with compute capability 9.0 or newer.

```shell
cmake -S native -B native/build \
  -DBUILD_TESTING=ON \
  -DFLIGHT4S_BUILD_CUDA_INTEGRATION_TESTS=ON
cmake --build native/build --config Release
ctest --test-dir native/build -C Release --output-on-failure
```

## JNI Smoke Test

Pass the absolute native library path to the runtime test:

```shell
sbt -Dflight4s.cuda.native.path=/absolute/path/to/flight4s_cuda \
  "runtime/testOnly flight4s.runtime.cuda.CudaResourcesJniSuite flight4s.runtime.cuda.NvrtcCompilerSuite flight4s.runtime.cuda.internal.NativeCudaLauncherSuite"
```

Without this property, portable Scala test runs skip the JNI-dependent tests.
Native library packaging and platform classifiers remain future work.

Run all Scala and native-enabled tests with the same property:

```shell
sbt -Dflight4s.cuda.native.path=/absolute/path/to/flight4s_cuda test
```

The half-precision range test additionally reads `CUDA_PATH` as the toolkit
root and passes its `include` directory explicitly to NVRTC. That directory
must contain `cuda_fp16.h`. This is test/application configuration, not implicit
header discovery in the runtime compiler.

## C++ Language Standard

The native runtime and generated CUDA C++ target C++20. Generated artifacts
carry `--std=c++20` explicitly, together with the selected
`--gpu-architecture=compute_xy` target, rather than inheriting compiler
defaults. C++23 is deferred until NVCC and NVRTC support is consistent across
the supported Linux and Windows toolchains.
