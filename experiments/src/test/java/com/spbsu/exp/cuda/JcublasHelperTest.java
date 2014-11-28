package com.spbsu.exp.cuda;

import org.junit.Test;
import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import com.spbsu.exp.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.cuda.data.impl.FArrayVector;
import org.junit.Assert;

import java.util.Random;

/**
 * jmll
 * ksen
 * 14.October.2014 at 12:01
 */
public class JcublasHelperTest extends Assert {

  private final int M = 200;
  private final int K = 400;
  private final int N = 300;

  private final double epsilon = 1e-6;
  private final double epsilon2 = 1e-4;

  @Test
  public void testFMultAB() throws Exception {  // A[M x K] * B[K x N]
    final FMatrix A = getMatrix(M, K);
    final FMatrix B = getMatrix(K, N);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(K, copy(A.toArray()));
    final ColsVecArrayMx B2 = new ColsVecArrayMx(N, copy(B.toArray()));

    final Mx C2 = MxTools.multiply(A2, B2);
    final FMatrix C = JcublasHelper.fMult(A, B);

    assertTrue(compare(C2, C) < epsilon2);

    final FMatrix D = JcublasHelper.fMult(A, false, B, false);

    assertTrue(compare(C2, D) < epsilon2);
  }

  @Test
  public void testFMultATB() throws Exception {  // A[K x M] * B[K x N]
    final FMatrix A = getMatrix(K, M);
    final FMatrix B = getMatrix(K, N);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(M, copy(A.toArray()));
    final ColsVecArrayMx B2 = new ColsVecArrayMx(N, copy(B.toArray()));

    final Mx C2 = MxTools.multiply(MxTools.transpose(A2), B2);
    final FMatrix C = JcublasHelper.fMult(A, true, B, false);

    assertTrue(compare(C2, C) < epsilon2);
  }

  @Test
  public void testFMultABT() throws Exception {  // A[M x K] * B[N x K]
    final FMatrix A = getMatrix(M, K);
    final FMatrix B = getMatrix(N, K);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(K, copy(A.toArray()));
    final ColsVecArrayMx B2 = new ColsVecArrayMx(K, copy(B.toArray()));

    final Mx C2 = MxTools.multiply(A2, MxTools.transpose(B2));
    final FMatrix C = JcublasHelper.fMult(A, false, B, true);

    assertTrue(compare(C2, C) < epsilon2);
  }

  @Test
  public void testFMultATBT() throws Exception {  // A[K x M] * B[N x K]
    final FMatrix A = getMatrix(K, M);
    final FMatrix B = getMatrix(N, K);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(M, copy(A.toArray()));
    final ColsVecArrayMx B2 = new ColsVecArrayMx(K, copy(B.toArray()));

    final Mx C2 = MxTools.multiply(MxTools.transpose(A2), MxTools.transpose(B2));
    final FMatrix C = JcublasHelper.fMult(A, true, B, true);

    assertTrue(compare(C2, C) < epsilon2);
  }

  @Test
  public void testFDot() throws Exception {
    final int n = 1000;
    final FVector a = new FArrayVector(n);
    final ArrayVec a2 = new ArrayVec(n);
    final FVector b = new FArrayVector(n);
    final ArrayVec b2 = new ArrayVec(n);

    final Random random = new Random();
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      a.set(i, value);
      a2.set(i, value);
    }
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      b.set(i, value);
      b2.set(i, value);
    }

    final double c2 = VecTools.multiply(a2, b2);
    final float c = JcublasHelper.fDot(a, b);
    assertTrue(Math.abs(c2 - c) < 1e-4); // s and d comparison
  }

  @Test
  public void testFVMMult() throws Exception {
    final int m = 300;
    final int n = 200;
    final FMatrix A = new FArrayMatrix(m, n);
    final VecBasedMx A2 = new VecBasedMx(m, n);
    final FVector b = new FArrayVector(m);
    final ArrayVec b2 = new ArrayVec(m);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }
    for (int i = 0; i < m; i++) {
      final float value = random.nextFloat();
      b.set(i, value);
      b2.set(i, value);
    }

    final Vec c2 = MxTools.multiply(MxTools.transpose(A2), b2);
    final FVector c = JcublasHelper.fMult(b, A);

    double sum = 0;
    for (int i = 0; i < n; i++) {
      sum += Math.pow(c.get(i) - c2.get(i), 2);
    }
    assertTrue(Math.sqrt(sum / m) < 1e-4); // s and d comparison
  }

  @Test
  public void testFMVMult() throws Exception {
    final int m = 300;
    final int n = 200;
    final FMatrix A = new FArrayMatrix(m, n);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(m, n);
    final FVector b = new FArrayVector(n);
    final ArrayVec b2 = new ArrayVec(n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      b.set(i, value);
      b2.set(i, value);
    }

    final Vec c2 = MxTools.multiply(A2, b2);
    final FVector c = JcublasHelper.fMult(A, b);

    double sum = 0;
    for (int i = 0; i < n; i++) {
      sum += Math.pow(c.get(i) - c2.get(i), 2);
    }
    assertTrue(Math.sqrt(sum / m) < 1e-4); // s and d comparison
  }

  @Test
  public void testFVVMult() throws Exception {
    final int m = 300;
    final int n = 200;
    final FVector a = new FArrayVector(m);
    final ColsVecArrayMx a2 = new ColsVecArrayMx(m, 1);
    final FVector b = new FArrayVector(n);
    final ColsVecArrayMx b2 = new ColsVecArrayMx(1, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
        final float value = random.nextFloat();
        a.set(i, value);
        a2.set(i, 0, value);
    }
    for (int i = 0; i < n; i++) {
      final float value = random.nextFloat();
      b.set(i, value);
      b2.set(0, i, value);
    }

    final Mx c2 = MxTools.multiply(a2, b2);
    final FMatrix c = JcublasHelper.fMult(a, b);

    double sum = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        sum += Math.pow(c.get(i, j) - c2.get(i, j), 2);
      }
    }
    assertTrue(Math.sqrt(sum / m) < 1e-4); // s and d comparison
  }

  @Test
  public void testFSum() throws Exception {
    final int m = 500;
    final int n = 400;
    final FMatrix A = new FArrayMatrix(m, n);
    final VecBasedMx A2 = new VecBasedMx(m, n);
    final FMatrix B = new FArrayMatrix(m, n);
    final VecBasedMx B2 = new VecBasedMx(m, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        B.set(i, j, value);
        B2.set(i, j, value);
      }
    }

    final Mx C2 = VecTools.sum(A2, B2);
    final FMatrix C = JcublasHelper.fSum(A, B);

    double sum = 0;
    for (int i = 0; i < C.getRows(); i++) {
      for (int j = 0; j < C.getColumns(); j++) {
        sum += Math.pow(C.get(i, j) - C2.get(i, j), 2);
      }
    }
    assertTrue(Math.sqrt(sum / C.toArray().length) < 1e-4); // s and d comparison
  }

  @Test
  public void testFSubtract() throws Exception {
    final int m = 1000;
    final int n = 997;
    final FMatrix A = new FArrayMatrix(m, n);
    final VecBasedMx A2 = new VecBasedMx(m, n);
    final FMatrix B = new FArrayMatrix(m, n);
    final VecBasedMx B2 = new VecBasedMx(m, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        B.set(i, j, value);
        B2.set(i, j, value);
      }
    }

    final Mx C2 = VecTools.subtract(A2, B2);
    final FMatrix C = JcublasHelper.fSubtr(A, B);

    double sum = 0;
    for (int i = 0; i < C.getRows(); i++) {
      for (int j = 0; j < C.getColumns(); j++) {
        sum += Math.pow(C.get(i, j) - C2.get(i, j), 2);
      }
    }
    assertTrue(Math.sqrt(sum / C.toArray().length) < 1e-4); // s and d comparison
  }

  @Test
  public void testFScale() throws Exception {
    final int m = 1000;
    final int n = 997;
    final FMatrix A = new FArrayMatrix(m, n);
    final VecBasedMx A2 = new VecBasedMx(m, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }

    final float alpha = random.nextFloat();
    VecTools.scale(A2, alpha);
    JcublasHelper.fScale(A, alpha);

    double sum = 0;
    for (int i = 0; i < A.getRows(); i++) {
      for (int j = 0; j < A.getColumns(); j++) {
        sum += Math.pow(A.get(i, j) - A2.get(i, j), 2);
      }
    }
    assertTrue(Math.sqrt(sum / A.toArray().length) < 1e-4); // s and d comparison
  }

  // -------------------------------------------------------------------------------------------------------------------

  private double[] copy(final float[] array) {
    final double[] copy = new double[array.length];
    for (int i = 0; i < array.length; i++) {
      copy[i] = array[i];
    }
    return copy;
  }

  private FMatrix getMatrix(final int m, final int n) {
    final FMatrix mx = new FArrayMatrix(m, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        mx.set(i, j, random.nextFloat());
      }
    }
    return mx;
  }

  private float compare(final Mx expected, final FMatrix actual) {
    if (expected.rows() != actual.getRows() || expected.columns() != actual.getColumns()) {
      return Float.POSITIVE_INFINITY;
    }

    float error = 0;
    for (int i = 0; i < expected.rows(); i++) {
      for (int j = 0; j < expected.columns(); j++) {
        error += (float)Math.pow(expected.get(i, j) - actual.get(i, j), 2);
      }
    }
    return (float)Math.sqrt(error / (expected.rows() * expected.columns()));
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
