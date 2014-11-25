package com.spbsu.exp.cuda.data;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:55
 */
public interface FVector {

  float get(final int index);

  FVector set(final int index, final float value);

  float[] toArray();

  int getDimension();

}
