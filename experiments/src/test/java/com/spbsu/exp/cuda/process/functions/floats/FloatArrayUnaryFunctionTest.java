package com.spbsu.exp.cuda.process.functions.floats;

import org.junit.Test;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.process.functions.ArrayUnaryFunction;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import org.junit.Assert;

/**
 * jmll
 * ksen
 * 10.December.2014 at 00:46
 */
public class FloatArrayUnaryFunctionTest extends Assert {

  @Test
  public void testHeaviside() throws Exception {
    final ArrayUnaryFunction<float[]> heaviside = new HeavisideFA();
    final ArrayUnaryFunction<float[]> sigmoid = new SigmoidFA();

    final FVector x = new FArrayVector(new float[]{1, 2, 3});
    final FMatrix w = new FArrayMatrix(2, new float[]{4, 5, 6, 7});

    show(heaviside.f(x).toArray());
    show(sigmoid.f(x).toArray());
    show(heaviside.f(w).toArray());
    show(sigmoid.f(w).toArray());
  }

  private void show(final float[] array) {
    for (int i = 0; i < array.length; i++) {
      System.out.println(array[i] + "\t");
    }
    System.out.println();
  }

}
