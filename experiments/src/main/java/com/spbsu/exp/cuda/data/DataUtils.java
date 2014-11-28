package com.spbsu.exp.cuda.data;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;

/**
 * jmll
 * ksen
 * 25.November.2014 at 20:43
 */
public class DataUtils {

  public static float rmse(final @NotNull FMatrix A, final @NotNull FMatrix B) {
    return rmse(A.toArray(), B.toArray());
  }

  private static float rmse(final float[] a, final float[] b) {
    return (float)Math.sqrt(mse(a, b));
  }

  private static float mse(final float[] a, final float[] b) {
    float sum = 0;
    final int length = a.length;
    for (int i = 0; i < length; i++) {
      sum += (float)Math.pow(a[i] - b[i], 2);
    }
    return sum / length;
  }

  @NotNull
  public static FMatrix repeatAsColumns(final @NotNull FVector a, final int times) {
    return repeatAsColumns(a.toArray(), times);
  }

  @NotNull
  public static FMatrix repeatAsColumns(final @NotNull float[] a, final int times) {
    return new FArrayMatrix(a.length, repeat(a, times));
  }

  private static float[] repeat(final float[] source, final int times) {
    final int length = source.length;
    final float[] destination = new float[times * length];

    for (int i = 0; i < times; i++) {
      System.arraycopy(source, 0, destination, i * length, length);
    }
    return destination;
  }

}
