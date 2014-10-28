package com.spbsu.exp.dl.cuda.data.impl;

import com.spbsu.exp.dl.cuda.data.FMatrix;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:48
 */
public class FArrayMatrix implements FMatrix {

  private float[] data;

  private int rows;

  public FArrayMatrix(final int rows, final int columns) {
    data = new float[rows * columns];
    this.rows = rows;
  }

  public FArrayMatrix(final int rows, final float[] data) {
    this.data = data;
    this.rows = rows;
  }

  @Override
  public FMatrix set(final int i, final int j, final float value) {
    data[i + j * rows] = value;
    return this;
  }

  @Override
  public float get(final int i, final int j) {
    return data[i + j * rows];
  }

  @Override
  public float[] toArray() {
    return data;
  }

  public int getRows() {
    return rows;
  }

  public int getColumns() {
    return data.length / rows;
  }

}
