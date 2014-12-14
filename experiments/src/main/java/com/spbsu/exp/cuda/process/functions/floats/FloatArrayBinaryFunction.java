package com.spbsu.exp.cuda.process.functions.floats;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.data.ArrayBased;
import com.spbsu.exp.cuda.process.functions.ArrayBinaryFunction;

/**
 * jmll
 * ksen
 * 10.December.2014 at 01:16
 */
public abstract class FloatArrayBinaryFunction implements ArrayBinaryFunction<float[]> {

  @NotNull
  @Override
  public ArrayBased<float[]> f(final @NotNull ArrayBased<float[]> x, final @NotNull ArrayBased<float[]> z) {
    final float[] firstArgumentBase = x.toArray();
    final float[] secondArgumentBase = z.toArray();
    final float[] valueBase = new float[firstArgumentBase.length];

    map(firstArgumentBase, secondArgumentBase, valueBase);

    return x.reproduce(valueBase);
  }

  protected abstract void map(final float[] x, final float[] z, final float[] y);

}
