package com.spbsu.exp.cuda;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;
import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import jcuda.jcurand.curandRngType;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import org.jetbrains.annotations.NotNull;

import static jcuda.jcurand.JCurand.*;
import static jcuda.runtime.JCuda.*;

/**
 * jmll
 * ksen
 * 25.October.2014 at 21:36
 */
public class JcudaVectorInscale { //todo(ksen): reformat cp-ps

  static {
    JcudaHelper.warmUp();
  }

  public static FVector FHeaviside(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, a.get(i) < 0.f ? 0.f : 1.f);
    }
    return a;
  }

  public static FVector FSigmoid(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, 1.f / (1.f + (float) Math.exp(-a.get(i))));
    }
    return a;
  }

  public static FVector FExp(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, (float) Math.exp(a.get(i)));
    }
    return a;
  }

  public static FVector FTanh(final @NotNull FVector a) {
    for (int i = 0; i < a.getDimension(); i++) {
      a.set(i, (float) Math.tanh(a.get(i)));
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

  public static FMatrix fRndSigmoid(final @NotNull FMatrix A) {
    fRndSigmoid(A.toArray());
    return A;
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
    int x = (int) Math.pow(pow, 1. / 3.);
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
    int x = (int) Math.pow(pow, 1. / 3.);
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
    int x = (int) Math.pow(pow, 1. / 3.);
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

  private static void fRndSigmoid(final float[] ha) {
    final int length = ha.length;
    final float[] randomH = getRandom(length);

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
    JCudaDriver.cuModuleGetFunction(function, module, "fRndSigmoid");

    final CUdeviceptr original = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(original, length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(original, Pointer.to(ha), length * Sizeof.FLOAT);

    final CUdeviceptr randomD = new CUdeviceptr();
    JCudaDriver.cuMemAlloc(randomD, length * Sizeof.FLOAT);
    JCudaDriver.cuMemcpyHtoD(randomD, Pointer.to(randomH), length * Sizeof.FLOAT);

    Pointer kernelParameters = Pointer.to(
        Pointer.to(original),
        Pointer.to(randomD),
        Pointer.to(new int[]{length})
    );

    int pow = upper2pow(length);
    int x = (int) Math.pow(pow, 1. / 3.);
    int z = x > 1024 ? 1024 : x;
    int y = pow / (z * x);
    JCudaDriver.cuLaunchKernel(function,
        x, y, 1,
        z, 1, 1,
        0, null,
        kernelParameters, null
    );

    JCudaDriver.cuCtxSynchronize();

    JCudaDriver.cuMemcpyDtoH(Pointer.to(ha), original, length * Sizeof.FLOAT);

    JCudaDriver.cuMemFree(original);

    JCudaDriver.cuCtxDestroy(context);
  }

  private static float[] getRandom(final int size) {
    JCuda.setExceptionsEnabled(true);
    JCurand.setExceptionsEnabled(true);

    final curandGenerator generator = new curandGenerator();

    float host[] = new float[size];
    final Pointer device = new Pointer();
    cudaMalloc(device, size * Sizeof.FLOAT);

    curandCreateGenerator(generator, curandRngType.CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());

    curandGenerateUniform(generator, device, size);

    cudaMemcpy(Pointer.to(host), device, size * Sizeof.FLOAT, cudaMemcpyKind.cudaMemcpyDeviceToHost);

    curandDestroyGenerator(generator);
    cudaFree(device);

    return host;
  }

  private static int upper2pow(final int value) {
    return (int) Math.pow(2, 32 - Integer.numberOfLeadingZeros(value - 1));
  }

}
