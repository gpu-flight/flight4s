package flight4s.runtime.cuda.internal;

import java.io.File;
import java.nio.ByteBuffer;

final class CudaNativeBindings {
  private static final String EXPLICIT_LIBRARY_PATH =
      "flight4s.cuda.native.path";

  static {
    String explicitPath = System.getProperty(EXPLICIT_LIBRARY_PATH);
    if (explicitPath == null || explicitPath.isBlank()) {
      System.loadLibrary("flight4s_cuda");
    } else {
      System.load(new File(explicitPath).getAbsolutePath());
    }
  }

  private CudaNativeBindings() {}

  static native NativeCudaDriverResult launchKernel(
      long contextHandle,
      long functionHandle,
      long streamHandle,
      int gridX,
      int gridY,
      int gridZ,
      int blockX,
      int blockY,
      int blockZ,
      int dynamicSharedMemoryBytes,
      boolean usesCluster,
      int clusterX,
      int clusterY,
      int clusterZ,
      ByteBuffer argumentStorage,
      int[] argumentOffsets,
      byte[] argumentDescriptorCodes);

  static native NativeNvrtcResult compileCuda(
      byte[] sourceUtf8,
      byte[] programNameUtf8,
      byte[][] optionsUtf8);

  static native NativeNvrtcVersionResult queryNvrtcVersion();

  static native NativeCudaContextResult retainPrimaryContext(
      int deviceOrdinal);

  static native NativeCudaDriverResult releasePrimaryContext(
      int deviceOrdinal);

  static native NativeCudaDriverResult synchronizeContext(
      long contextHandle);

  static native NativeCudaDriverResult loadPtx(
      long contextHandle,
      byte[] ptx);

  static native NativeCudaDriverResult unloadModule(
      long contextHandle,
      long moduleHandle);

  static native NativeCudaDriverResult resolveFunction(
      long contextHandle,
      long moduleHandle,
      byte[] functionNameUtf8);

  static native NativeCudaFunctionAttributesResult queryFunctionAttributes(
      long contextHandle,
      long functionHandle);

  static native NativeCudaDriverResult createStream(
      long contextHandle,
      int flags);

  static native NativeCudaDriverResult destroyStream(
      long contextHandle,
      long streamHandle);

  static native NativeCudaDriverResult synchronizeStream(
      long contextHandle,
      long streamHandle);

  static native NativeCudaDriverResult createEvent(
      long contextHandle,
      int flags);

  static native NativeCudaDriverResult destroyEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult recordEvent(
      long contextHandle,
      long eventHandle,
      long streamHandle);

  static native NativeCudaEventQueryResult queryEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult synchronizeEvent(
      long contextHandle,
      long eventHandle);

  static native NativeCudaDriverResult waitForEvent(
      long contextHandle,
      long streamHandle,
      long eventHandle);

  static native NativeCudaDriverResult createBlasHandle(
      long contextHandle);

  static native NativeCudaDriverResult destroyBlasHandle(
      long contextHandle,
      long blasHandle);

  static native NativeCudaDriverResult executeGemmF32(
      long contextHandle,
      long blasHandle,
      long streamHandle,
      int m,
      int n,
      int k,
      float alpha,
      float beta,
      long aAddress,
      long bAddress,
      long cAddress,
      long workspaceAddress,
      long workspaceBytes);

  static native NativeCudaPinnedMemoryResult allocatePinnedMemory(
      long contextHandle,
      long sizeBytes);

  static native NativeCudaDriverResult freePinnedMemory(
      long contextHandle,
      long hostAddress);

  static native NativeCudaDriverResult allocateDeviceMemory(
      long contextHandle,
      long sizeBytes);

  static native NativeCudaDriverResult freeDeviceMemory(
      long contextHandle,
      long deviceAddress);

  static native NativeCudaDriverResult copyHostToDevice(
      long contextHandle,
      long deviceAddress,
      long deviceOffsetBytes,
      ByteBuffer source);

  static native NativeCudaDriverResult copyHostToDeviceAsync(
      long contextHandle,
      long deviceAddress,
      long deviceOffsetBytes,
      ByteBuffer source,
      long streamHandle);

  static native NativeCudaDriverResult copyDeviceToHost(
      long contextHandle,
      long deviceAddress,
      long deviceOffsetBytes,
      ByteBuffer destination);

  static native NativeCudaDriverResult copyDeviceToHostAsync(
      long contextHandle,
      long deviceAddress,
      long deviceOffsetBytes,
      ByteBuffer destination,
      long streamHandle);
}
