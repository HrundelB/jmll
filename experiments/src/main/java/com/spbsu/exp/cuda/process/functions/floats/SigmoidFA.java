package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:34
 */
public class SigmoidFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.f / (1.f + (float)Math.exp(-x[i]));
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    float argument;
    for (int i = 0; i < x.length; i++) {
      argument = x[i];
      y[i] = argument - argument * argument;
    }
  }

}
