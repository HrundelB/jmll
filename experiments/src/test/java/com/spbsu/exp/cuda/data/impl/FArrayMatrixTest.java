package com.spbsu.exp.cuda.data.impl;

import org.junit.Test;
import com.spbsu.exp.cuda.data.FMatrix;
import com.spbsu.exp.cuda.data.FVector;
import org.junit.Assert;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:56
 */
public class FArrayMatrixTest extends Assert {

  private final int dim = 3;

  @Test
  public void testCreate() throws Exception {
    final FMatrix A = new FArrayMatrix(dim, dim);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.set(i, j, i * dim + j);
      }
    }

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        assertTrue(A.get(i, j) == i * dim + j);
      }
    }
    final float[] array = A.toArray();
    assertTrue(array[0] == 0);     assertTrue(array[1] == 3);     assertTrue(array[2] == 6);
    assertTrue(array[3] == 1);     assertTrue(array[4] == 4);     assertTrue(array[5] == 7);
    assertTrue(array[6] == 2);     assertTrue(array[7] == 5);     assertTrue(array[8] == 8);
  }

  @Test
  public void testGetColumn() throws Exception {
    final FMatrix A = new FArrayMatrix(dim, dim);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.set(i, j, i * dim + j);
      }
    }

    for (int i = 0; i < dim; i++) {
      final FVector column = A.getColumn(i);

      for (int j = 0; j < dim; j++) {
        assertTrue(j * dim + i == column.get(j));
      }
    }
  }

  @Test
  public void testSetColumn() throws Exception {
    final FMatrix A = new FArrayMatrix(dim, dim);

    final float[] a1 = new float[]{1, 2, 3};
    final float[] a2 = new float[]{4, 5, 6};
    final float[] a3 = new float[]{7, 8, 9};

    A.setColumn(0, a1);
    A.setColumn(1, a2);
    A.setColumn(2, a3);

    assertTrue(a1[0] == A.get(0, 0));     assertTrue(a2[0] == A.get(0, 1));     assertTrue(a3[0] == A.get(0, 2));
    assertTrue(a1[1] == A.get(1, 0));     assertTrue(a2[1] == A.get(1, 1));     assertTrue(a3[1] == A.get(1, 2));
    assertTrue(a1[2] == A.get(2, 0));     assertTrue(a2[2] == A.get(2, 1));     assertTrue(a3[2] == A.get(2, 2));
  }

  @Test
  public void testSetColumnVec() throws Exception {
    final FMatrix A = new FArrayMatrix(dim, dim);

    final FVector a1 = new FArrayVector(new float[]{1, 2, 3});
    final FVector a2 = new FArrayVector(new float[]{4, 5, 6});
    final FVector a3 = new FArrayVector(new float[]{7, 8, 9});

    A.setColumn(0, a1);
    A.setColumn(1, a2);
    A.setColumn(2, a3);

    assertTrue(a1.get(0) == A.get(0, 0));  assertTrue(a2.get(0) == A.get(0, 1));  assertTrue(a3.get(0) == A.get(0, 2));
    assertTrue(a1.get(1) == A.get(1, 0));  assertTrue(a2.get(1) == A.get(1, 1));  assertTrue(a3.get(1) == A.get(1, 2));
    assertTrue(a1.get(2) == A.get(2, 0));  assertTrue(a2.get(2) == A.get(2, 1));  assertTrue(a3.get(2) == A.get(2, 2));
  }

  @Test
  public void testGetColumnsRange() throws Exception {
    final FMatrix A = new FArrayMatrix(dim, dim);

    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        A.set(i, j, i * dim + j);
      }
    }

    final FMatrix B = A.getColumnsRange(1, 2);

    assertTrue(1 == B.get(0, 0));  assertTrue(2 == B.get(0, 1));
    assertTrue(4 == B.get(1, 0));  assertTrue(5 == B.get(1, 1));
    assertTrue(7 == B.get(2, 0));  assertTrue(8 == B.get(2, 1));
  }

  @Test
  public void testSetPieceOfColumn() throws Exception {
    final FMatrix A = new FArrayMatrix(3, new float[]{
        1, 2, 3, 1, 2, 3, 1, 2, 3
    });

    A.setPieceOfColumn(0, 0, new FArrayVector(new float[]{3, 2}));
    A.setPieceOfColumn(1, 0, new FArrayVector(new float[]{3, 2}));
    A.setPieceOfColumn(2, 0, new FArrayVector(new float[]{3, 2}));

    System.out.println(A);
  }
}
