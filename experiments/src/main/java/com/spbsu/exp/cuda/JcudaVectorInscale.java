package com.spbsu.exp.cuda;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 25.October.2014 at 21:36
 */
public class JcudaVectorInscale { //todo(ksen): reformat cp-ps

  public static FVector FHeaviside(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, a.get(i) < 0.f ? 0.f : 1.f);
    }
    return a;
  }

  public static FVector FSigmoid(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, 1.f / (1.f + (float)Math.exp(-a.get(i))));
    }
    return a;
  }

  public static FVector FExp(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, (float)Math.exp(a.get(i)));
    }
    return a;
  }

  public static FVector FTanh(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, (float)Math.tanh(a.get(i)));
    }
    return a;
  }

  public static FVector FIdentity(final int size) {
    final float[] a = new float[size];
    for (int i = 0; i < size; i++) {
      a[i] = 1.f;
    }
    return new FArrayVector(a);
  }

  public static FMatrix fSigmoid(final @NotNull FMatrix A) {
    fSigmoid(A.toArray());
    return A;
  }

  public static FVector fSigmoid(final @NotNull FVector a) {
    fSigmoid(a.toArray());
    return a;
  }

  public static FMatrix fTanh(final @NotNull FMatrix A) {
    fTanh(A.toArray());
    return A;
  }

  public static FVector fTanh(final @NotNull FVector a) {
    fTanh(a.toArray());
    return a;
  }

  public static FMatrix fExp(final @NotNull FMatrix A) {
    fExp(A.toArray());
    return A;
  }

  public static FVector fExp(final @NotNull FVector a) {
    fExp(a.toArray());
    return a;
  }

  public static FMatrix fRepeatAsRow(final @NotNull FVector a, final int times) {
    throw new UnsupportedOperationException();
  }

  public static void fExp(final float[] ha) {
    final String cudaKernel = JcudaHelper.getPtx("VectorInscale.ptx");
    JCudaDriver.setExceptionsEnabled(true);

    JCudaDriver.cuInit(0);
    final CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    final CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    final CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoad(module, cudaKernel);

    final CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, "fExp");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.FLOAT);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int)Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.FLOAT);

    JCudaDriver.cuMemFree(da);

    JCudaDriver.cuCtxDestroy(context);
  }

  private static void fSigmoid(final float[] ha) {
    final String cudaKernel = JcudaHelper.getPtx("VectorInscale.ptx");
    JCudaDriver.setExceptionsEnabled(true);

    JCudaDriver.cuInit(0);
    final CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    final CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    final CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoad(module, cudaKernel);

    final CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, "fSigmoid");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.FLOAT);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int)Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.FLOAT);

    JCudaDriver.cuMemFree(da);

    JCudaDriver.cuCtxDestroy(context);
  }

  private static void fTanh(final float[] ha) {
    final String cudaKernel = JcudaHelper.getPtx("VectorInscale.ptx");
    JCudaDriver.setExceptionsEnabled(true);

    JCudaDriver.cuInit(0);
    final CUdevice device = new CUdevice();
    JCudaDriver.cuDeviceGet(device, 0);
    final CUcontext context = new CUcontext();
    JCudaDriver.cuCtxCreate(context, 0, device);

    final CUmodule module = new CUmodule();
    JCudaDriver.cuModuleLoad(module, cudaKernel);

    final CUfunction function = new CUfunction();
    JCudaDriver.cuModuleGetFunction(function, module, "fTanh");

    final int length = ha.length;

    final CUdeviceptr da = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(da, length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(da, Pointer.to(ha), length * Sizeof.FLOAT);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(da),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int)Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), da, length * Sizeof.FLOAT);

    JCudaDriver.cuMemFree(da);

    JCudaDriver.cuCtxDestroy(context);
  }

  private static int upper2pow(final int value) {
    return (int)Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
