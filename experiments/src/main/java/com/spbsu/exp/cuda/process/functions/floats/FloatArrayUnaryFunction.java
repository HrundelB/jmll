package com.spbsu.exp.cuda.process.functions.floats;

import org.jetbrains.annotations.NotNull;
import com.spbsu.exp.cuda.process.functions.ArrayUnaryFunction;
import com.spbsu.exp.cuda.data.ArrayBased;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:12
 */
public abstract class FloatArrayUnaryFunction implements ArrayUnaryFunction<float[]> {

  @NotNull
  @Override
  public ArrayBased<float[]> f(final @NotNull ArrayBased<float[]> x) {
    final float[] argumentBase = x.toArray();
    final float[] valueBase = new float[argumentBase.length];

    map(argumentBase, valueBase);

    return x.reproduce(valueBase);
  }

  @NotNull
  @Override
  public ArrayBased<float[]> df(final @NotNull ArrayBased<float[]> x) {
    final float[] argumentBase = x.toArray();
    final float[] valueBase = new float[argumentBase.length];

    dMap(argumentBase, valueBase);

    return x.reproduce(valueBase);
  }

  protected abstract void map(final float[] x, final float[] y);

  protected abstract void dMap(final float[] x, final float[] y);

}
