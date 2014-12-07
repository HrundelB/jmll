package com.spbsu.exp.cuda.data;

import gnu.trove.list.TIntList;
import org.jetbrains.annotations.NotNull;

/**
 * jmll
 * ksen
 * 23.October.2014 at 21:55
 */
public interface FVector {

  float get(final int index);

  FVector set(final int index, final float value);

  @NotNull
  FVector getRange(final @NotNull TIntList indexes);

  float[] toArray();

  int getDimension();

}
