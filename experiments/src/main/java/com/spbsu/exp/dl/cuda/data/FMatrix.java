package com.spbsu.exp.dl.cuda.data;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:46
 */
public interface FMatrix {

  FMatrix set(final int i, final int j, final float value);

  float get(final int i, final int j);

  // Column representation
  float[] toArray();

  int getRows();

  int getColumns();

}
