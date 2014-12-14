package com.spbsu.exp.cuda.process.functions.floats;

import java.util.Random;

/**
 * jmll
 * ksen
 * 09.December.2014 at 23:53
 */
public class RandomSigmoidFA extends FloatArrayUnaryFunction {

  private Random random = new Random();

  @Override
  protected void map(final float[] x, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = 1.f / (1.f + (float)Math.exp(-x[i])) > random.nextFloat() ? 1.f : 0.f;
    }
  }

  @Override
  protected void dMap(final float[] x, final float[] y) {
    throw new UnsupportedOperationException();
  }

}
