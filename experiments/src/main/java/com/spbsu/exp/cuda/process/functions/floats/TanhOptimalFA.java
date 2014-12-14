package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:02
 */
public class TanhOptimalFA extends FloatArrayUnaryFunction {

  @Override
  protected void map(final float[] x, final float[] y) {
    final float alpha = 2.f / 3.f;
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.7159f * (float)Math.tanh(alpha * x[i]);
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    final float alpha = 1.7159f * 2.f / 3.f;
    final float beta = 1.f / (float)Math.pow(1.7159, 2);
    for (int i = 0; i < x.length; i++) {
      y[i] = alpha * (1.f - beta * (float)Math.pow(x[i], 2));
    }
  }

}
