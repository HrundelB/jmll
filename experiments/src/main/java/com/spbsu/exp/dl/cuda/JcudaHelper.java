package com.spbsu.exp.dl.cuda;

import com.spbsu.commons.system.RuntimeUtils;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import org.jetbrains.annotations.NotNull;

import java.io.*;
import java.lang.reflect.Field;
import java.util.Arrays;

import static jcuda.driver.JCudaDriver.*;

/**
 * jmll
 * ksen
 * 16.October.2014 at 11:35
 */
public class JcudaHelper { //todo(ksenon): from jar (dll to tmp)

  public static final String RESOURCES_DIR = "experiments/src/main/resources/jcuda/";
  public static final String LIBS_DIR = RESOURCES_DIR + "libs/";
  public static final String KERNELS_DIR = RESOURCES_DIR + "kernels/cus";
  public static final String PTX_DIR = RESOURCES_DIR + "kernels/ptxs/";

  static {
    try {
      final Field usrPathsField = ClassLoader.class.getDeclaredField("usr_paths");
      usrPathsField.setAccessible(true);

      final String[] paths = (String[]) usrPathsField.get(null);
      final String[] newPaths = Arrays.copyOf(paths, paths.length + 1);
      newPaths[newPaths.length - 1] = LIBS_DIR;

      usrPathsField.set(null, newPaths);
    } catch (NoSuchFieldException | IllegalAccessException e) {
      throw new RuntimeException(e);
    }
  }

  public static String getPtx(final @NotNull String ptxFileName) {
    return PTX_DIR + ptxFileName;
  }

  public static void rebuildAll() {
    final String[] cus = new File(KERNELS_DIR).list();

    for (final String cu : cus) {
      buildKernel(cu, cuNameToPtx(cu));
    }
  }

  private static String cuNameToPtx(final String cuFileName) {
    final int extensionPoint = cuFileName.lastIndexOf('.');
    if (extensionPoint == -1) {
      throw new RuntimeException("Wrong extension of the file: " + cuFileName);
    }
    return PTX_DIR + cuFileName.substring(0, extensionPoint + 1) + "ptx";
  }

  public static String buildKernel(final @NotNull String cuFileName) {
    final String cuFilePath = KERNELS_DIR + cuFileName;
    final String ptxFilePath = PTX_DIR + cuNameToPtx(cuFilePath);

    buildKernel(cuFilePath, ptxFilePath);

    return ptxFilePath;
  }

  public static void buildKernel(final @NotNull String cuFilePath, final @NotNull String ptxFilePath) {
    buildKernel(new File(cuFilePath), new File(ptxFilePath));
  }

  public static void buildKernel(final @NotNull File cuFile, final @NotNull File ptxFile) {
    final String command = buildCommand(cuFile, ptxFile);

    final int exitCode;
    final String stdErr;
    final String stdOut;
    try {
      final Process process = Runtime.getRuntime().exec(command);

      stdErr = streamToString(process.getErrorStream());
      stdOut = streamToString(process.getInputStream());
      exitCode = process.waitFor();
    }
    catch (Exception e) {
      Thread.currentThread().interrupt();
      throw new RuntimeException("Interrupted while waiting for nvcc output", e);
    }

    if (exitCode != 0) {
      System.out.println("nvcc ended with exit code: " + exitCode);
      System.out.println("Std Err:\n" + stdErr);
      System.out.println("Std Out:\n" + stdOut);
      throw new RuntimeException("Could not create .ptx file: " + stdErr);
    }
  }

  private static String buildCommand(final File cuFile, final File ptxFile) {
    if (!cuFile.isFile()) {
      throw new RuntimeException(cuFile.getAbsolutePath() + " isn't file.", new IllegalArgumentException());
    }

    return new StringBuilder()
                   .append("nvcc ")
                   .append("-m ").append(RuntimeUtils.getArchDataModel()).append(' ')
                   .append("-ptx ").append(cuFile.getAbsolutePath()).append(' ')
                   .append("-o ").append(ptxFile.getAbsolutePath())
                   .toString()
    ;
  }

  private static String streamToString(final InputStream inputStream) {
    final StringBuilder builder = new StringBuilder();

    try (final LineNumberReader reader = new LineNumberReader(new InputStreamReader(inputStream))) {
      final char[] buffer = new char[8192];

      int read;
      while ((read = reader.read(buffer)) != -1) {
        builder.append(buffer, 0, read);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return builder.toString();
  }

  public static void warmUp() {
    cuInit(0);
    final CUdevice device = new CUdevice();
    cuDeviceGet(device, 0);
    final CUcontext context = new CUcontext();
    cuCtxCreate(context, 0, device);

    final int N = 1_000;
    final int size = Sizeof.DOUBLE * N;
    final double[] hData = new double[N];

    final CUdeviceptr dData = new CUdeviceptr();
    cuMemAlloc(dData, size);

    cuMemcpyHtoD(dData, Pointer.to(hData), size);
    cuMemcpyDtoH(Pointer.to(hData), dData, size);
    cuMemFree(dData);
  }

}
