package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:50
 */
public class TanhFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = (float)Math.tanh(x[i]);
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = (float)Math.pow(1.f / (float)Math.cosh(x[i]), 2.);
    }
  }

}
