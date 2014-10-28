package com.spbsu.exp.dl.cuda;

import com.spbsu.commons.math.vectors.*;
import com.spbsu.commons.math.vectors.impl.basis.MxBasisImpl;
import com.spbsu.commons.math.vectors.impl.mx.ColsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.RowsVecArrayMx;
import com.spbsu.commons.math.vectors.impl.mx.SparseMx;
import com.spbsu.commons.math.vectors.impl.mx.VecBasedMx;
import com.spbsu.commons.math.vectors.impl.vectors.ArrayVec;
import com.spbsu.commons.seq.regexp.Matcher;
import com.spbsu.exp.dl.cuda.data.FMatrix;
import com.spbsu.exp.dl.cuda.data.FVector;
import com.spbsu.exp.dl.cuda.data.impl.FArrayMatrix;
import com.spbsu.exp.dl.cuda.data.impl.FArrayVector;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;
import java.util.Random;

/**
 * jmll
 * ksen
 * 14.October.2014 at 12:01
 */
public class JcublasHelperTest extends Assert {

  @Test
  public void testMult() throws Exception {
    final int m = 1000;
    final int k = 2000;
    final int n = 1000;

    final ColsVecArrayMx A = new ColsVecArrayMx(m, k);
    final ColsVecArrayMx B = new ColsVecArrayMx(k, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        A.set(i, j, random.nextDouble());
      }
    }
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        B.set(i, j, random.nextDouble());
      }
    }

    long begin = System.currentTimeMillis();
    final Mx C = MxTools.multiply(A, B);
    System.out.println("CPU: " + (System.currentTimeMillis() - begin));
    begin = System.currentTimeMillis();
    final Mx D = JcublasHelper.mult(A, B);
    System.out.println("GPU: " + (System.currentTimeMillis() - begin));

    assertTrue(C.rows() == D.rows());
    assertTrue(C.columns() == D.columns());

    double sum = 0;
    for (int i = 0; i < C.rows(); i++) {
      for (int j = 0; j < C.columns(); j++) {
        sum += Math.pow(C.get(i, j) - D.get(i, j), 2);
      }
    }
    System.out.println("RMSE: " + Math.sqrt(sum / C.toArray().length));
  }

  @Test
  public void testSpeed() throws Exception {
    for (int a = 1; a < 1000; a++) {
      final ColsVecArrayMx A = new ColsVecArrayMx(a, a);
      final ColsVecArrayMx B = new ColsVecArrayMx(a, a);

      final Random random = new Random();
      for (int i = 0; i < a; i++) {
        for (int j = 0; j < a; j++) {
          A.set(i, j, random.nextDouble());
        }
      }
      for (int i = 0; i < a; i++) {
        for (int j = 0; j < a; j++) {
          B.set(i, j, random.nextDouble());
        }
      }

      long begin = System.currentTimeMillis();
      MxTools.multiply(A, B);
      System.out.println(a + " CPU: " + (System.currentTimeMillis() - begin));
      begin = System.currentTimeMillis();
      JcublasHelper.mult(A, B);
      System.out.println(a + " GPU: " + (System.currentTimeMillis() - begin));
    }
  }

  @Test
  public void testFMult() throws Exception {
    final int m = 100;
    final int k = 200;
    final int n = 300;
    final FMatrix A = new FArrayMatrix(m, k);
    final ColsVecArrayMx A2 = new ColsVecArrayMx(m, k);
    final FMatrix B = new FArrayMatrix(k, n);
    final ColsVecArrayMx B2 = new ColsVecArrayMx(k, n);

    final Random random = new Random();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < k; j++) {
        final float value = random.nextFloat();
        A.set(i, j, value);
        A2.set(i, j, value);
      }
    }
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        final float value = random.nextFloat();
        B.set(i, j, value);
        B2.set(i, j, value);
      }
    }

    final Mx C2 = MxTools.multiply(A2, B2);
    final FMatrix C = JcublasHelper.fMult(A, B);

    double sum = 0;
    for (int i = 0; i < C.getRows(); i++) {
      for (int j = 0; j < C.getColumns(); j++) {
        sum += Math.pow(C.get(i, j) - C2.get(i, j), 2);
      }
    }
    assertTrue(Math.sqrt(sum / C.toArray().length) < 1e-4); // s and d comparison
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

  @Test
  public void testMxEquals() throws Exception {
    final Mx[] mxes = getMxes();
    final Mx[] clone = mxes.clone();

    final String pattern = "%-20s %-20s %-20s %-20s %-20s";
    String[] line = new String[]{
        "Mx \\ Mx",
        mxes[0].getClass().getSimpleName(),
        mxes[1].getClass().getSimpleName(),
        mxes[2].getClass().getSimpleName(),
        mxes[3].getClass().getSimpleName()
    };
    System.out.println(String.format(pattern, line));
    System.out.println();

    for (int i = 0; i < 4; i++) {
      line[0] = mxes[i].getClass().getSimpleName();
      for (int j = 0; j < 4; j++) {
        line[j + 1] = mxes[i].equals(clone[j]) + "";
      }
      System.out.println(String.format(pattern, line));
    }
    System.out.println();

    line = new String[]{
        "Mx \\ Mx",
        mxes[0].getClass().getSimpleName(),
        mxes[1].getClass().getSimpleName(),
        mxes[2].getClass().getSimpleName(),
        mxes[3].getClass().getSimpleName()
    };
    System.out.println(String.format(pattern, line));
    System.out.println();

    for (int i = 0; i < 4; i++) {
      line[0] = mxes[i].getClass().getSimpleName();
      for (int j = 0; j < 4; j++) {
        line[j + 1] = compare(mxes[i], clone[j]) + "";
      }
      System.out.println(String.format(pattern, line));
    }
  }

  @Test
  public void testMxRepresentations() throws Exception {
    final Mx[] mxes = getMxes();
    for (Mx mx : mxes) {
      System.out.println(mx.getClass().getSimpleName() + "\t" + Arrays.toString(mx.toArray()));
    }
    System.out.println("T:\n");
    for (Mx mx : mxes) {
      System.out.println(mx.getClass().getSimpleName() + "\t" + Arrays.toString(MxTools.transpose(mx).toArray()));
    }
  }

  private Mx[] getMxes() {
    final Mx vbm = new VecBasedMx(2, 2)
        .set(0, 0, 0)
        .set(0, 1, 1)
        .set(1, 0, 2)
        .set(1, 1, 3);
    final Mx cvam = new ColsVecArrayMx(new Vec[]{
        new ArrayVec(0, 1),
        new ArrayVec(2, 3)}
    );
    final Mx rcam = new RowsVecArrayMx(new Vec[]{
        new ArrayVec(0, 1),
        new ArrayVec(2, 3)
    });
    final Mx sm = new SparseMx<MxBasis>(new MxBasisImpl(2, 2))
        .set(0, 0, 0)
        .set(0, 1, 1)
        .set(1, 0, 2)
        .set(1, 1, 3);

    return new Mx[]{vbm, cvam, rcam, sm};
  }

  private boolean compare(Mx A, Mx B) {
    if (A.rows() != B.rows() || A.columns() != B.columns()) {
      return false;
    }
    for (int i = 0; i < A.rows(); i++) {
      for (int j = 0; j < A.columns(); j++) {
        if (A.get(i, j) != B.get(i, j)) {
          return false;
        }
      }
    }
    return true;
  }

}
