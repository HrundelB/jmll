package com.spbsu.exp.cuda;

import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import org.junit.Test;
import com.spbsu.commons.math.vectors.Vec;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import org.junit.Assert;

import java.util.Random;

/**
 * jmll
 * ksen
 * 25.October.2014 at 21:59
 */
public class JcudaVectorInscaleTest extends Assert { // speedup after 2.5M (cuz ptx reading)

  @Test
  public void testFSigmoid() throws Exception {
    final int n = 3000000;
    FVector a = new FArrayVector(n);
    ArrayVec a2 = new ArrayVec(n);

    final Random random = new Random();
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      a.set(i, value);
      a2.set(i, value);
    }

    for (int i = 0; i < n; i++) {
      a2.set(i, 1. / (1. + Math.exp(-a2.get(i))));
    }
    JcudaVectorInscale.fSigmoid(a);

    compare(a, a2);
  }

  @Test
  public void testFExp() throws Exception {
    final int n = 3000000;
    FVector a = new FArrayVector(n);
    ArrayVec a2 = new ArrayVec(n);

    final Random random = new Random();
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      a.set(i, value);
      a2.set(i, value);
    }

    for (int i = 0; i < n; i++) {
      a2.set(i, Math.exp(a2.get(i)));
    }
    JcudaVectorInscale.fExp(a);

    compare(a, a2);
  }

  @Test
  public void testName() throws Exception {
    FMatrix A = new FArrayMatrix(3, new float[]{-10, -1, -0.00001f, 0, 0.01f, 10});

    show(JcudaVectorInscale.fRndSigmoid(A));
  }

  private void compare(final FVector a, final Vec b) {
    double sum = 0;
    for (int i = 0; i < a.getDimension(); i++) {
      sum += Math.pow(a.get(i) - b.get(i), 2);
    }
    assertTrue(Math.sqrt(sum / a.getDimension()) < 1e-4); // s and d comparison
  }

  private void show(final FMatrix A) {
    for (int i = 0; i < A.getRows(); i++) {
      for (int j = 0; j < A.getColumns(); j++) {
        System.out.printf(A.get(i, j) + "\t");
      }
      System.out.println();
    }
    System.out.println();
  }

}
