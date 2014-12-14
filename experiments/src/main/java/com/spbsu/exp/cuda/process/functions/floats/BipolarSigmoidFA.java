package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 14.December.2014 at 00:23
 */
public class BipolarSigmoidFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    float value;
    for (int i = 0; i < x.length; i++) {
      value = x[i];
      y[i] = (1.f - (float)Math.exp(-value)) / (1.f + (float)Math.exp(-value));
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    float value;
    for (int i = 0; i < x.length; i++) {
      value = x[i];
      y[i] = 2 * ((float)Math.exp(value) / (float)Math.pow(Math.exp(value) + 1., 2));
    }
  }

}
