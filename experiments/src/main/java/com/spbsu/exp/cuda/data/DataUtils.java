package com.spbsu.exp.cuda.data;

import com.spbsu.exp.cuda.data.impl.FArrayVector;
import gnu.trove.list.array.TIntArrayList;
import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;

import java.util.Random;

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

  @NotNull
  public static FMatrix extendAsBottomRow(final @NotNull FMatrix A, final @NotNull FVector b) {
    final int rows = A.getRows();
    final int columns = A.getColumns();
    final FMatrix extended = new FArrayMatrix(rows + 1, columns);
    for (int i = 0; i < columns; i++) {
      extended.setPieceOfColumn(i, 0, A.getColumn(i));
      extended.set(rows, i, b.get(i));
    }
    return extended;
  }

  @NotNull
  public static FMatrix contractBottomRow(final @NotNull FMatrix A) {
    final int rows = A.getRows() - 1;
    final int columns = A.getColumns();
    final FMatrix contracted = new FArrayMatrix(rows, columns);
    for (int i = 0; i < columns; i++) {
      contracted.setPieceOfColumn(i, 0, rows, A.getColumn(i));
    }
    return contracted;
  }

  @NotNull
  public static FVector once(final int size) {
    final float[] data = new float[size];
    for (int i = 0; i < size; i++) {
      data[i] = 1.f;
    }
    return new FArrayVector(data);
  }

  @NotNull
  public static FMatrix once(final int rows, final int columns) {
    final int dim = rows * columns;
    final float[] data = new float[dim];
    for (int i = 0; i < dim; i++) {
      data[i] = 1.f;
    }
    return new FArrayMatrix(rows, data);
  }

  public static TIntArrayList randomPermutations(final int size) {
    final Random random = new Random();
    final TIntArrayList list = new TIntArrayList(size);

    for (int i = 0; i < size; i++) {
      list.add(i);
    }
    list.shuffle(random);

    return list;
  }

}
