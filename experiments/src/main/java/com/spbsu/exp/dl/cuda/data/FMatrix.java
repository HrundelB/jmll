package com.spbsu.exp.dl.cuda.data;

import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:46
 */
public interface FMatrix {

  FMatrix set(final int i, final int j, final float value);

  float get(final int i, final int j);

  FVector getColumn(final int j);

  void setColumn(final int j, final @NotNull FVector column);

  void setColumn(final int j, final float[] column);

  // Column representation
  float[] toArray();

  int getRows();

  int getColumns();

}
