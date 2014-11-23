package com.spbsu.exp.dl.cuda.data.impl;

import com.spbsu.exp.dl.cuda.data.FMatrix;
import com.spbsu.exp.dl.cuda.data.FVector;
import org.junit.Assert;
import org.junit.Test;

import java.util.Arrays;

/**
 * jmll
 * ksen
 * 22.October.2014 at 22:56
 */
public class FArrayMatrixTest extends Assert {

  @Test
  public void testCreate() throws Exception {
    final int dim = 3;
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
    final int dim = 3;
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

}
