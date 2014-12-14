package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:58
 */
public class IdenticalFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    System.arraycopy(x, 0, y, 0, x.length);
  }

  @Override
  protected void dMap(final float[] x, float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.f;
    }
  }

}
