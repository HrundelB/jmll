package com.spbsu.exp.dl.cuda.data.impl;

import com.spbsu.exp.dl.cuda.data.FMatrix;
import com.spbsu.exp.dl.cuda.data.FVector;
import org.jetbrains.annotations.NotNull;

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
  public FVector getColumn(final int j) {
    final float[] destination = new float[rows];
    System.arraycopy(data, rows * j, destination, 0, rows);
    return new FArrayVector(destination);
  }

  @Override
  public void setColumn(final int j, final @NotNull FVector column) {
    setColumn(j, column.toArray());
  }

  @Override
  public void setColumn(int j, float[] column) {
    System.arraycopy(column, 0, data, rows * j, rows);
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
