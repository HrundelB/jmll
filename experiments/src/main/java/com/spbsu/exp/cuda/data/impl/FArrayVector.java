package com.spbsu.exp.cuda.data.impl;

import com.spbsu.exp.cuda.data.FVector;
import gnu.trove.list.TIntList;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:56
 */
public class FArrayVector implements FVector {

  private float[] data;

  public FArrayVector(final float[] data) {
    this.data = data;
  }

  public FArrayVector(final int dimension) {
    data = new float[dimension];
  }

  @Override
  public float get(final int index) {
    return data[index];
  }

  @Override
  public FVector set(int index, float value) {
    data[index] = value;
    return this;
  }

  @NotNull
  @Override
  public FVector getRange(final @NotNull TIntList indexes) {
    final int size = indexes.size();
    final float[] subVector = new float[size];
    for (int i = 0; i < size; i++) {
      subVector[i] = data[indexes.get(i)];
    }
    return new FArrayVector(subVector);
  }

  @Override
  public float[] toArray() {
    return data;
  }

  @Override
  public int getDimension() {
    return data.length;
  }

  @Override
  public String toString() {
    final StringBuilder builder = new StringBuilder();
    for (int i = 0; i < data.length; i++) {
      builder.append(data[i]).append(' ');
    }
    return builder.append('\n').toString();
  }

}
