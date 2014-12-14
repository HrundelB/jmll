package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:32
 */
public class HeavisideFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = x[i] < 0.f ? 0.f : 1.f;
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    throw new UnsupportedOperationException();
  }

}
