package com.spbsu.exp.cuda.process.functions.floats;

/**
 * jmll
 * ksen
 * 13.December.2014 at 12:39
 */
public class HadamardFA extends FloatArrayBinaryFunction {

  @Override
  protected void map(final float[] x, final float[] z, final float[] y) {
    for (int i = 0; i < x.length; i++) {
      y[i] = x[i] * z[i];
    }
  }

}
